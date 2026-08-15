"""Draft model with DFlash-style KV injection, supporting discrete-mask and
ELF-style continuous-flow drafting.

Forward paths:
  forward (train)  : queries are A packed blocks of S positions per sequence;
                     keys/values are [target context features ; block tokens].
                     Bidirectional within a block, causal w.r.t. context
                     features (ctx position < block anchor position), no
                     cross-block attention. SDPA with a boolean mask.
  infer_step       : inference path, mirrors dflash incremental decoding with
                     a DynamicCache holding projected context-feature KV.

State-dict layout matches z-lab/Qwen3-*-DFlash checkpoints (layers.*, fc,
hidden_norm, norm) so released weights load directly for harness validation.
Flow extras (time_mlp, mode_emb, self_cond_proj, flow_head, emb_mu/sigma
buffers) are additive.

Flow input processing (ELF): normalized state z concat self-cond x_hat' ->
self_cond_proj -> de-normalize to embedding scale -> add time+mode cond
(non-anchor slots) -> trunk. Anchor slot always gets the clean embedding.
"""
import math
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)

from cflow.flow import MODE_DENOISE


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, n_heads, S, dh]; cos/sin: [B, S, dh]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DraftAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        hs = config.hidden_size
        self.q_proj = nn.Linear(hs, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hs, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hs, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hs, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv(self, hidden: torch.Tensor, ctx: Optional[torch.Tensor]):
        B, Q = hidden.shape[:2]
        q = self.q_proj(hidden).view(B, Q, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, Q, dh]
        k_noise = self.k_proj(hidden).view(B, Q, self.num_kv_heads, self.head_dim)
        v_noise = self.v_proj(hidden).view(B, Q, self.num_kv_heads, self.head_dim)
        if ctx is not None:
            M = ctx.shape[1]
            k_ctx = self.k_proj(ctx).view(B, M, self.num_kv_heads, self.head_dim)
            v_ctx = self.v_proj(ctx).view(B, M, self.num_kv_heads, self.head_dim)
            k = torch.cat([k_ctx, k_noise], dim=1)
            v = torch.cat([v_ctx, v_noise], dim=1)
        else:
            k, v = k_noise, v_noise
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_blocks(self, hidden, ctx, cos_q, sin_q, cos_ctx, sin_ctx, attn_mask):
        B, Q = hidden.shape[:2]
        M = ctx.shape[1]
        q, k, v = self._qkv(hidden, ctx)
        q = apply_rope(q, cos_q, sin_q)
        k_ctx = apply_rope(k[:, :, :M], cos_ctx, sin_ctx)
        k_noise = apply_rope(k[:, :, M:], cos_q, sin_q)
        k = torch.cat([k_ctx, k_noise], dim=2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scaling, enable_gqa=True
        )
        out = out.transpose(1, 2).reshape(B, Q, -1)
        return self.o_proj(out)

    def forward_infer(self, hidden, ctx, cos_q, sin_q, cos_ctx, sin_ctx, past_key_values):
        B, Q = hidden.shape[:2]
        M = ctx.shape[1] if ctx is not None else 0
        q, k, v = self._qkv(hidden, ctx)
        q = apply_rope(q, cos_q, sin_q)
        if M > 0:
            k_ctx = apply_rope(k[:, :, :M], cos_ctx, sin_ctx)
            k_noise = apply_rope(k[:, :, M:], cos_q, sin_q)
            k = torch.cat([k_ctx, k_noise], dim=2)
        else:
            k = apply_rope(k, cos_q, sin_q)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scaling, enable_gqa=True
        )  # block attends to all ctx + itself, bidirectional
        out = out.transpose(1, 2).reshape(B, Q, -1)
        return self.o_proj(out)


class DraftLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = DraftAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden, attn_fn):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = attn_fn(self.self_attn, hidden)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class CFlowDraftModel(nn.Module):
    """KV-injected draft. mode='mask' reproduces DFlash; mode='flow' is the
    ELF-adapted continuous drafter."""

    def __init__(self, config: Qwen3Config, mode: str = "mask"):
        super().__init__()
        self.config = config
        self.mode = mode
        self.layers = nn.ModuleList([DraftLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.target_layer_ids = (getattr(config, "target_layer_ids", None) or
                                 build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = getattr(config, "block_size", 16)
        self.mask_token_id = getattr(config, "mask_token_id", None)

        if mode == "flow":
            hs = config.hidden_size
            self.time_mlp = nn.Sequential(nn.Linear(256, hs), nn.SiLU(), nn.Linear(hs, hs))
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)
            self.mode_emb = nn.Embedding(2, hs)
            nn.init.zeros_(self.mode_emb.weight)
            self.self_cond_proj = nn.Linear(2 * hs, hs, bias=False)
            with torch.no_grad():  # init = identity on z, zero on self-cond
                self.self_cond_proj.weight.zero_()
                self.self_cond_proj.weight[:, :hs] = torch.eye(hs)
            self.flow_head = nn.Linear(hs, hs, bias=True)
            self.register_buffer("emb_mu", torch.tensor(0.0), persistent=True)
            self.register_buffer("emb_sigma", torch.tensor(1.0), persistent=True)

        if mode == "rcf":
            hs = config.hidden_size
            # TE(t): zero-init weights; bias later set to embed(mask_token) so
            # that at init the t=0 input slot is exactly e_mask (U1 parity)
            self.time_mlp = nn.Sequential(nn.Linear(256, hs), nn.SiLU(), nn.Linear(hs, hs))
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)
            self.w_sc = nn.Linear(hs, hs, bias=False)
            nn.init.zeros_(self.w_sc.weight)

    def init_rcf_bias(self, mask_emb: torch.Tensor):
        with torch.no_grad():
            self.time_mlp[-1].bias.copy_(mask_emb.to(self.time_mlp[-1].bias.dtype))

    def build_rcf_inputs(self, x, sc, t, anchor_emb):
        """x, sc: [B, A, S, D] raw-embedding-space states (slot 0 ignored);
        t: [B, A]; anchor_emb: [B, A, D]. Returns [B, A*S, D]."""
        B, A, S_, D = x.shape
        w_dtype = self.time_mlp[0].weight.dtype
        te = self.time_mlp(timestep_embedding(t, 256).to(w_dtype))       # [B, A, D]
        slot = x + te.unsqueeze(2).to(x.dtype) + self.w_sc(sc)
        inp = torch.cat([anchor_emb.unsqueeze(2), slot[:, :, 1:, :]], dim=2)
        return inp.reshape(B, A * S_, D)

    def set_embedding_stats(self, mu: float, sigma: float):
        self.emb_mu.fill_(mu)
        self.emb_sigma.fill_(sigma)

    # ---------------- input construction (flow) ----------------

    def build_flow_inputs(self, z, sc, anchor_emb, S):
        """z: [B, A, S, D] normalized states; sc: same shape or None;
        anchor_emb: [B, A, D] clean anchor embeddings (embedding scale).
        Returns [B, A*S, D] trunk inputs at embedding scale."""
        B, A, S_, D = z.shape
        sc = torch.zeros_like(z) if sc is None else sc
        zt = self.self_cond_proj(torch.cat([z, sc], dim=-1))
        x_in = zt * self.emb_sigma + self.emb_mu
        x_in = torch.cat([anchor_emb.unsqueeze(2), x_in[:, :, 1:, :]], dim=2)
        return x_in.reshape(B, A * S_, D)

    def cond_vector(self, t, mode_ids):
        """t: [B, A] in [0,1]; mode_ids: [B, A] in {0,1} -> [B, A, D]."""
        w_dtype = self.time_mlp[0].weight.dtype
        cond = self.time_mlp(timestep_embedding(t, 256).to(w_dtype))
        cond = cond + self.mode_emb(mode_ids).to(cond.dtype)
        return cond

    def fuse_ctx(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    def rope(self, ref: torch.Tensor, positions: torch.Tensor):
        cos, sin = self.rotary_emb(ref, positions)
        return cos.to(ref.dtype), sin.to(ref.dtype)

    # ---------------- training path ----------------

    def forward(self, block_inputs, target_hidden, anchors, t=None, mode_ids=None,
                flow_z=None, flow_sc=None, anchor_emb=None,
                rcf_x=None, rcf_sc=None, rcf_t=None):
        """Training entry point. Pass ONE of: block_inputs ([B, A*S, D],
        embedding scale — mask mode), flow_z/flow_sc (ELF flow mode), or
        rcf_x/rcf_sc/rcf_t (RCF mode). Input construction happens inside so
        trainable input modules stay within the DDP forward. Returns (h, z_hat)."""
        if flow_z is not None:
            block_inputs = self.build_flow_inputs(flow_z, flow_sc, anchor_emb, self.block_size)
        elif rcf_x is not None:
            block_inputs = self.build_rcf_inputs(rcf_x, rcf_sc, rcf_t, anchor_emb)
            t = None  # time enters via TE inside build_rcf_inputs only
        h = self.forward_blocks(block_inputs, target_hidden, anchors, t, mode_ids)
        z_hat = self.flow_head(h) if self.mode == "flow" else None
        return h, z_hat

    def forward_blocks(self, block_inputs, target_hidden, anchors, t=None, mode_ids=None):
        B, Q, D = block_inputs.shape
        S = self.block_size
        A = Q // S
        M = target_hidden.shape[1]
        device = block_inputs.device

        ctx = self.fuse_ctx(target_hidden)

        offs = torch.arange(S, device=device)
        q_pos = (anchors.unsqueeze(-1) + offs).reshape(B, Q)
        ctx_pos = torch.arange(M, device=device).unsqueeze(0).expand(B, M)

        cos_q, sin_q = self.rope(block_inputs, q_pos)
        cos_ctx, sin_ctx = self.rope(block_inputs, ctx_pos)

        blk_id = torch.arange(A, device=device).repeat_interleave(S)
        ctx_allow = ctx_pos.unsqueeze(1) < anchors[:, blk_id].unsqueeze(-1)   # [B, Q, M]
        blk_allow = (blk_id.unsqueeze(0) == blk_id.unsqueeze(1)).expand(B, Q, Q)
        attn_mask = torch.cat([ctx_allow, blk_allow], dim=-1).unsqueeze(1)

        hidden = block_inputs
        if t is not None:
            cond = self.cond_vector(t, mode_ids)                              # [B, A, D]
            cond = cond.repeat_interleave(S, dim=1)                           # [B, Q, D]
            not_anchor = (offs != 0).repeat(A).view(1, Q, 1).to(cond.dtype)
            hidden = hidden + cond * not_anchor
        for layer in self.layers:
            hidden = layer(hidden, lambda attn, h: attn.forward_blocks(
                h, ctx, cos_q, sin_q, cos_ctx, sin_ctx, attn_mask))
        return self.norm(hidden)

    # ---------------- inference path ----------------

    def infer_step(
        self,
        block_inputs: torch.Tensor,                 # [B, S, D] embedding-scale
        block_positions: torch.Tensor,              # [B, S]
        new_target_hidden: Optional[torch.Tensor],  # [B, m, L*D] or None (cached)
        new_ctx_positions: Optional[torch.Tensor],  # [B, m]
        past_key_values: DynamicCache,
        t: Optional[torch.Tensor] = None,           # [B]
        mode_ids: Optional[torch.Tensor] = None,    # [B]
    ) -> torch.Tensor:
        ctx = self.fuse_ctx(new_target_hidden) if new_target_hidden is not None else None
        cos_q, sin_q = self.rope(block_inputs, block_positions)
        cos_ctx = sin_ctx = None
        if ctx is not None:
            cos_ctx, sin_ctx = self.rope(block_inputs, new_ctx_positions)
        hidden = block_inputs
        if t is not None:
            cond = self.cond_vector(t.unsqueeze(-1), mode_ids.unsqueeze(-1))  # [B, 1, D]
            keep = torch.ones(1, hidden.shape[1], 1, device=hidden.device, dtype=cond.dtype)
            keep[:, 0] = 0.0
            hidden = hidden + cond * keep
        for layer in self.layers:
            hidden = layer(hidden, lambda attn, h: attn.forward_infer(
                h, ctx, cos_q, sin_q, cos_ctx, sin_ctx, past_key_values))
        return self.norm(hidden)

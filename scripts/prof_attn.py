import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from cflow.modeling import CFlowDraftModel, build_target_layer_ids

device = "cuda:0"
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
for p in target.parameters():
    p.requires_grad_(False)

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
cfg.num_hidden_layers = 5
cfg.num_target_layers = 36
cfg.block_size = 16
cfg.mask_token_id = 151669
cfg.target_layer_ids = build_target_layer_ids(36, 5)
draft = CFlowDraftModel(cfg, mode="mask").to(device)

B, A, S, M = 8, 64, 16, 1024
D = cfg.hidden_size
Q = A * S
ids = torch.randint(1000, 100000, (B, M), device=device)
anchors = torch.randint(100, M - 20, (B, A), device=device)


def timeit(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


# 1. target forward (no attn mask)
with torch.no_grad():
    t_target = timeit(lambda: target.model(input_ids=ids, output_hidden_states=True))
out = target.model(input_ids=ids, output_hidden_states=True)
feats = torch.cat([out.hidden_states[l + 1] for l in cfg.target_layer_ids], dim=-1)

inputs = torch.randn(B, Q, D, device=device, dtype=torch.bfloat16)

# 2. draft forward+backward (current bool-mask SDPA)
def full_fb():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, _ = draft(inputs, feats, anchors)
    h.sum().backward()
t_draft_fb = timeit(full_fb, 3)

# 3. head + CE fwd+bwd only
h0 = torch.randn(B, Q, D, device=device, dtype=torch.bfloat16, requires_grad=True)
labels = torch.randint(0, 100000, (B * Q,), device=device)
def ce_fb():
    logits = F.linear(h0, target.lm_head.weight)
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels)
    loss.backward()
t_ce = timeit(ce_fb, 3)

# 4. raw SDPA variants for one attention call shape
Hq, Hkv, dh = 32, 8, 128
q = torch.randn(B, Hq, Q, dh, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, Hkv, M + Q, dh, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn_like(k).requires_grad_(True)
blk = torch.arange(A, device=device).repeat_interleave(S)
ctx_allow = torch.arange(M, device=device).view(1, 1, M) < anchors[:, blk].unsqueeze(-1)
blk_allow = (blk.unsqueeze(0) == blk.unsqueeze(1)).expand(B, Q, Q)
mask_bool = torch.cat([ctx_allow, blk_allow], dim=-1).unsqueeze(1)

def sdpa_bool():
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bool, enable_gqa=True)
    o.sum().backward()
t_bool = timeit(sdpa_bool, 3)

mask_add = torch.zeros_like(mask_bool, dtype=torch.bfloat16).masked_fill(~mask_bool, torch.finfo(torch.bfloat16).min)
def sdpa_add():
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_add, enable_gqa=True)
    o.sum().backward()
t_add = timeit(sdpa_add, 3)

# 5. flex attention
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

anc = anchors
def mask_mod(b, h, qi, ki):
    qb = qi // S
    in_ctx = ki < M
    ctx_ok = ki < anc[b, qb]
    blk_ok = (ki - M) // S == qb
    return torch.where(in_ctx, ctx_ok, blk_ok)

bm = create_block_mask(mask_mod, B, 1, Q, M + Q, device=device)
def flex_eager():
    o = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)
    o.sum().backward()
t_flex = timeit(flex_eager, 3)

flex_c = torch.compile(flex_attention)
def flex_comp():
    o = flex_c(q, k, v, block_mask=bm, enable_gqa=True)
    o.sum().backward()
t_flexc = timeit(flex_comp, 3)

print(f"target_fwd(M={M})={t_target:.0f}ms | draft5L_fb_boolmask={t_draft_fb:.0f}ms | head+CE_fb={t_ce:.0f}ms")
print(f"1-layer SDPA fb: bool={t_bool:.0f}ms add={t_add:.0f}ms flex_eager={t_flex:.0f}ms flex_compiled={t_flexc:.0f}ms")

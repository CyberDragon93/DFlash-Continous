"""DDP training for the draft model: mode=mask (DFlash baseline reproduction)
or mode=flow (ELF-style continuous flow matching, arXiv 2605.10938).

Run:
  torchrun --nproc_per_node=4 -m cflow.train --data data/train_raw.jsonl \
      --mode flow --name flow_v1
"""
import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM

from cflow.data.dataset import BucketBatchSampler, SpecDataset, collate
from cflow.flow import (
    MODE_DECODE,
    MODE_DENOISE,
    FlowConfig,
    emb_stats,
    endpoint_from_logits,
    sample_logit_normal,
)
from cflow.modeling import CFlowDraftModel, build_target_layer_ids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--mode", choices=["mask", "flow", "rcf"], default="mask")
    ap.add_argument("--draft-layers", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--mask-token-id", type=int, default=151669)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=7.0, help="position loss decay")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps cap (smoke)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", default="")
    # flow (ELF) params
    ap.add_argument("--denoise-prob", type=float, default=0.8)
    ap.add_argument("--t-mean", type=float, default=-1.5)
    ap.add_argument("--t-std", type=float, default=0.8)
    ap.add_argument("--p-mean", type=float, default=0.8)
    ap.add_argument("--p-std", type=float, default=0.8)
    ap.add_argument("--noise-scale", type=float, default=2.0)
    ap.add_argument("--decode-noise-scale", type=float, default=1.0)
    ap.add_argument("--no-self-cond", action="store_true")
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    # rcf params
    ap.add_argument("--warm-start", default="", help="'zlab' or path: load trunk weights before training")
    ap.add_argument("--p-roll", type=float, default=0.5)
    ap.add_argument("--t1-lo", type=float, default=0.30)
    ap.add_argument("--t1-hi", type=float, default=0.85)
    ap.add_argument("--depth2-prob", type=float, default=0.15)
    ap.add_argument("--sc-keep", type=float, default=0.5)
    ap.add_argument("--top-m", type=int, default=64)
    ap.add_argument("--T-r", type=float, default=1.0)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--decode-pure-prob", type=float, default=0.1)
    # rcf v2: stochastic source for draft diversity (best-of-N drafting)
    ap.add_argument("--pass1-noise-prob", type=float, default=0.0,
                    help="fraction of pass-1 blocks whose x0 is embedding-scale noise instead of 0")
    ap.add_argument("--pass1-noise-scale", type=float, default=0.7,
                    help="noise std as a multiple of the embedding-matrix RMS")
    # fork/commit: teach the drafter to CONTINUE a committed in-block prefix
    ap.add_argument("--commit-prob", type=float, default=0.0,
                    help="per-block prob of committing a true prefix of the block")
    ap.add_argument("--commit-len-max", type=int, default=12)
    ap.add_argument("--commit-len-mean", type=float, default=5.0)
    return ap.parse_args()


def sample_commit(B, A, S, offs, args, device):
    """Returns (keep_clean [B,A,S] bool, w [B,A,S] float): committed offsets
    keep true-token embeddings and get zero loss weight; position decay
    restarts after the commit."""
    sel = torch.rand(B, A, device=device) < args.commit_prob
    u = torch.rand(B, A, device=device).clamp(min=1e-6)
    geo = (torch.log(u) / torch.log(torch.tensor(1.0 - 1.0 / args.commit_len_mean, device=device))).long() + 1
    c = torch.where(sel, geo.clamp(1, args.commit_len_max), torch.zeros_like(geo))
    o = offs.view(1, 1, S)
    keep_clean = o <= c[..., None]                      # offsets 1..c clean (offset 0 = anchor anyway)
    rel = (o - c[..., None] - 1).float()
    w = torch.exp(-rel.clamp(min=0.0) / args.gamma) * (o > c[..., None]).float()
    w[:, :, 0] = 0.0
    return keep_clean, w


def load_warm_start(draft, spec, device):
    """Load trunk weights from 'zlab', a safetensors dir, or one of our
    ckpt.pt files."""
    import glob
    import os
    if spec.endswith(".pt"):
        ck = torch.load(spec, map_location="cpu", weights_only=False)
        sd = ck["model"]
    else:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        local = snapshot_download("z-lab/Qwen3-4B-DFlash-b16") if spec == "zlab" else spec
        sd = {}
        for fp in glob.glob(os.path.join(local, "*.safetensors")):
            sd.update(load_file(fp))
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return len(sd), [m for m in missing]


def build_blocks(input_ids, seq_lens, prompt_lens, A, S):
    """Sample A anchors per sequence in the response region; gather block
    tokens. Each block offset >= 1 predicts its own token; offset 0 = anchor."""
    B, M = input_ids.shape
    device = input_ids.device
    low = prompt_lens
    high = torch.maximum(seq_lens - 2, low)
    span = (high - low + 1).float()
    anchors = low.unsqueeze(1) + (torch.rand(B, A, device=device) * span.unsqueeze(1)).long()
    offs = torch.arange(S, device=device)
    pos = anchors.unsqueeze(-1) + offs
    valid = pos < seq_lens.view(B, 1, 1)
    block_tokens = input_ids.gather(1, pos.clamp(max=M - 1).view(B, -1)).view(B, A, S)
    labels = block_tokens.masked_fill(~valid, -100)
    labels[:, :, 0] = -100
    return anchors, block_tokens, labels, valid


def weighted_mean(per_tok, sel, offs_weight):
    """per_tok, sel: [B, A, S]; offs_weight: [S]. Mean weighted by decay and
    restricted to sel."""
    w = offs_weight.view(1, 1, -1) * sel.float()
    return (per_tok * w).sum() / w.sum().clamp(min=1.0)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed * 1000 + rank)
    is_main = rank == 0

    out_dir = os.path.join(args.out_root, args.name)
    os.makedirs(out_dir, exist_ok=True)

    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    embed = target.model.embed_tokens
    lm_head_w = target.lm_head.weight
    D = target.config.hidden_size
    S = args.block_size
    A = args.anchors

    cfg = AutoConfig.from_pretrained(args.target)
    cfg.num_hidden_layers = args.draft_layers
    cfg.num_target_layers = target.config.num_hidden_layers
    cfg.block_size = S
    cfg.mask_token_id = args.mask_token_id
    cfg.target_layer_ids = build_target_layer_ids(cfg.num_target_layers, args.draft_layers)
    draft = CFlowDraftModel(cfg, mode=args.mode).to(device)
    tl_ids = draft.target_layer_ids

    fcfg = FlowConfig(denoise_prob=args.denoise_prob, t_mean=args.t_mean, t_std=args.t_std,
                      p_mean=args.p_mean, p_std=args.p_std, noise_scale=args.noise_scale,
                      decode_noise_scale=args.decode_noise_scale, self_cond=not args.no_self_cond)
    mu, sigma = emb_stats(embed.weight)
    if args.mode == "flow":
        draft.set_embedding_stats(mu, sigma)
    if args.warm_start:
        n_loaded, missing = load_warm_start(draft, args.warm_start, device)
        if is_main:
            print(f"warm-start: loaded {n_loaded} tensors; new modules: {missing}", flush=True)
    if args.mode == "rcf":
        draft.init_rcf_bias(embed.weight[args.mask_token_id].detach().float())

    ds = SpecDataset(args.data, max_len=args.max_len, limit=args.limit_samples)
    sampler = BucketBatchSampler(ds.lengths, args.micro_batch, world=world, rank=rank, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                        num_workers=4, pin_memory=True)

    model = DDP(draft, device_ids=[local]) if world > 1 else draft
    opt = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, args.beta2))

    steps_per_epoch = len(loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    start_epoch, gstep = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        draft.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, gstep = ck["epoch"] + 1, ck["gstep"]
        if is_main:
            print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    offs_weight = torch.exp(-(torch.arange(S, device=device).float() - 1.0) / args.gamma)
    offs_weight[0] = 0.0
    mask_emb = embed.weight[args.mask_token_id].detach().clone()

    ckpt_cfg = {**vars(args), "target_layer_ids": tl_ids, "emb_mu": mu, "emb_sigma": sigma}
    if is_main:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({**ckpt_cfg, "world": world, "total_steps": total_steps}, f, indent=2)
        logf = open(os.path.join(out_dir, "log.jsonl"), "a")

    def save_ckpt(epoch):
        ck = {"model": draft.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "gstep": gstep, "cfg": ckpt_cfg}
        tmp = os.path.join(out_dir, "ckpt.pt.tmp")
        torch.save(ck, tmp)
        os.rename(tmp, os.path.join(out_dir, "ckpt.pt"))
        print(f"saved checkpoint at epoch {epoch}", flush=True)

    prof = bool(os.environ.get("CFLOW_TIME"))

    def tick():
        if prof:
            torch.cuda.synchronize()
        return time.time()

    done = False
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)  # drop any non-step-aligned tail grads
        t_last = time.time()
        t_end = time.time()
        for it, batch in enumerate(loader):
            t_data = tick() - t_end
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            seq_lens = batch["seq_lens"].to(device, non_blocking=True)
            prompt_lens = batch["prompt_lens"].to(device, non_blocking=True)
            B, M = input_ids.shape

            t0 = tick()
            with torch.no_grad():
                # no attention_mask: right padding is causally safe (valid
                # tokens never attend to the padded tail) and passing a mask
                # forces HF off the fast flash-causal path (~2-3x slower)
                out = target.model(input_ids=input_ids, output_hidden_states=True)
                feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)
            t_tgt = tick() - t0

            anchors, block_tokens, labels, pos_valid = build_blocks(input_ids, seq_lens, prompt_lens, A, S)
            emb = embed(block_tokens)                                         # [B,A,S,D] bf16
            valid = labels != -100

            if args.mode == "mask":
                inputs = emb.clone()
                inputs[:, :, 1:, :] = mask_emb.to(emb.dtype)
                if args.commit_prob > 0:
                    offs_t = torch.arange(S, device=device)
                    keep_clean, w_commit = sample_commit(B, A, S, offs_t, args, device)
                    # committed offsets: true-token embeddings, but only where
                    # the position is in-range (overhang stays mask)
                    kc = keep_clean & pos_valid
                    inputs = torch.where(kc[..., None], emb, inputs)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, _ = model(inputs.view(B, A * S, D), feats, anchors)
                    logits = F.linear(h, lm_head_w)
                per_ce = F.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), labels.view(-1),
                    reduction="none", ignore_index=-100).view(B, A, S)
                if args.commit_prob > 0:
                    wm = w_commit * valid.float()
                    loss_ce = (per_ce * wm).sum() / wm.sum().clamp(min=1.0)
                else:
                    loss_ce = weighted_mean(per_ce, valid, offs_weight)
                loss = loss_ce
                loss_mse_val = 0.0
            elif args.mode == "rcf":
                anchor_emb = emb[:, :, 0, :]
                zeros_x = torch.zeros_like(emb)
                t0 = torch.zeros(B, A, device=device)
                x0_in = zeros_x
                if args.pass1_noise_prob > 0:
                    # stochastic source: per-block noise for draft diversity
                    sigma0 = embed.weight.float().pow(2).mean().sqrt()
                    noisy = (torch.rand(B, A, device=device) < args.pass1_noise_prob)
                    eps0 = torch.randn_like(emb, dtype=torch.float32) * (args.pass1_noise_scale * sigma0)
                    x0_in = (eps0 * noisy[..., None, None].float()).to(emb.dtype)
                commit_kwargs = {}
                w_commit = None
                if args.commit_prob > 0:
                    offs_t = torch.arange(S, device=device)
                    keep_clean, w_commit = sample_commit(B, A, S, offs_t, args, device)
                    kc = keep_clean & pos_valid
                    commit_kwargs = dict(rcf_commit_mask=kc, rcf_commit_emb=emb)
                # PASS 1 (t=0): identical task to the mask baseline (+ noisy-source blocks)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h1, _ = model(None, feats, anchors,
                                  rcf_x=x0_in, rcf_sc=zeros_x, rcf_t=t0, anchor_emb=anchor_emb,
                                  **commit_kwargs)
                    logits1 = F.linear(h1, lm_head_w)
                per_ce1 = F.cross_entropy(
                    logits1.float().view(-1, logits1.shape[-1]), labels.view(-1),
                    reduction="none", ignore_index=-100).view(B, A, S)
                if w_commit is not None:
                    wm = w_commit * valid.float()
                    loss_ce = (per_ce1 * wm).sum() / wm.sum().clamp(min=1.0)
                else:
                    loss_ce = weighted_mean(per_ce1, valid, offs_weight)
                loss = loss_ce
                loss_mse_val = 0.0  # reused as the pass-2 CE slot below
                # PASS 2 (rollout branch): inference states from own predictions.
                # Branch decisions MUST be identical across DDP ranks (a rank
                # that skips pass-2 deadlocks the others in allreduce), so use
                # a host RNG seeded only by (seed, epoch, it).
                step_rng = __import__("random").Random(args.seed * 7919 + epoch * 1_000_003 + it)
                if step_rng.random() < args.p_roll:
                    with torch.no_grad():
                        x1 = endpoint_from_logits(logits1.detach(), embed.weight,
                                                  args.T_r, args.top_m).view(B, A, S, D)
                        t1 = args.t1_lo + (args.t1_hi - args.t1_lo) * torch.rand(B, A, device=device)
                        x = t1[..., None, None] * x1
                        sc_src = x1
                        if step_rng.random() < args.depth2_prob:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                h1b, _ = model(None, feats, anchors,
                                               rcf_x=x.to(emb.dtype), rcf_sc=x1.to(emb.dtype),
                                               rcf_t=t1, anchor_emb=anchor_emb)
                                logits1b = F.linear(h1b, lm_head_w)
                            x1b = endpoint_from_logits(logits1b, embed.weight,
                                                       args.T_r, args.top_m).view(B, A, S, D)
                            t2 = t1 + 0.1 + (0.9 - t1 - 0.1).clamp(min=0.0) * torch.rand(B, A, device=device)
                            t2 = t2.clamp(max=0.9)
                            dt = (t2 - t1)[..., None, None]
                            omt = (1.0 - t1)[..., None, None].clamp(min=1e-3)
                            x = x + dt / omt * (x1b - x)
                            t1 = t2
                            sc_src = x1b
                        keep = (torch.rand(B, A, device=device) < args.sc_keep).float()[..., None, None]
                        sc = sc_src * keep
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h2, _ = model(None, feats, anchors,
                                      rcf_x=x.to(emb.dtype), rcf_sc=sc.to(emb.dtype),
                                      rcf_t=t1, anchor_emb=anchor_emb)
                        logits2 = F.linear(h2, lm_head_w)
                    per_ce2 = F.cross_entropy(
                        logits2.float().view(-1, logits2.shape[-1]), labels.view(-1),
                        reduction="none", ignore_index=-100).view(B, A, S)
                    loss_roll = weighted_mean(per_ce2, valid, offs_weight)
                    loss = loss + loss_roll
                    loss_mse_val = loss_roll.item()
            else:
                x = (emb.float() - mu) / sigma                                # [B,A,S,D]
                # overhang slots: no clean signal (match inference, where those
                # slots hold flow states, never pad-token embeddings)
                x = x * pos_valid[..., None].float()
                denoise_sel = torch.rand(B, A, device=device) < fcfg.denoise_prob
                t_d = sample_logit_normal((B, A), fcfg.t_mean, fcfg.t_std, device)
                eps_d = torch.randn_like(x) * fcfg.noise_scale
                z_den = t_d[..., None, None] * x + (1 - t_d[..., None, None]) * eps_d
                p_tok = sample_logit_normal((B, A, S), fcfg.p_mean, fcfg.p_std, device)
                # p=0 atom: train the decode branch on content-free inputs so
                # the K=1 (decode-from-noise) arm is in-distribution
                if args.decode_pure_prob > 0:
                    p_tok = p_tok * (torch.rand(B, A, S, device=device) >= args.decode_pure_prob).float()
                eps_dec = torch.randn_like(x) * fcfg.decode_noise_scale
                z_dec = p_tok[..., None] * x + (1 - p_tok[..., None]) * eps_dec
                den4 = denoise_sel[..., None, None].float()
                z = den4 * z_den + (1 - den4) * z_dec
                t_all = torch.where(denoise_sel, t_d, torch.ones_like(t_d))
                mode_ids = torch.where(denoise_sel,
                                       torch.full_like(denoise_sel, MODE_DENOISE, dtype=torch.long),
                                       torch.full_like(denoise_sel, MODE_DECODE, dtype=torch.long))
                anchor_emb = emb[:, :, 0, :]

                sc = None
                if fcfg.self_cond:
                    sc_sel = denoise_sel & (torch.rand(B, A, device=device) < 0.5)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        _, xhat0 = model(None, feats, anchors, t_all, mode_ids,
                                         flow_z=z.to(emb.dtype), anchor_emb=anchor_emb)
                    sc = xhat0.float().view(B, A, S, D) * sc_sel[..., None, None].float()
                    sc = sc.to(emb.dtype)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, z_hat = model(None, feats, anchors, t_all, mode_ids,
                                     flow_z=z.to(emb.dtype), flow_sc=sc, anchor_emb=anchor_emb)
                    logits = F.linear(h, lm_head_w)

                # denoise branch: velocity MSE; decode branch: CE through
                # frozen lm_head. Common-denominator normalization so the
                # realized gradient allocation follows the 80/20 branch split
                # (ELF Alg 1) instead of reweighting each branch to parity.
                one_minus_t = (1 - t_all)[..., None, None].clamp(min=1e-3)
                v_pred = (z_hat.float().view(B, A, S, D) - z) / one_minus_t
                v_tgt = x - eps_d
                per_mse = (v_pred - v_tgt).pow(2).mean(-1)                    # [B,A,S]
                per_ce = F.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), labels.view(-1),
                    reduction="none", ignore_index=-100).view(B, A, S)
                w_all = offs_weight.view(1, 1, -1)
                m_den = (valid & denoise_sel[..., None]).float() * w_all
                m_dec = (valid & (~denoise_sel)[..., None]).float() * w_all
                denom = (m_den + m_dec).sum().clamp(min=1.0)
                loss_mse = (per_mse * m_den).sum() / denom
                loss_ce = (per_ce * m_dec).sum() / denom
                loss = args.mse_weight * loss_mse + args.ce_weight * loss_ce
                loss_mse_val = loss_mse.item()

            t1 = tick()
            (loss / args.accum).backward()
            if prof and (it % 10 == 0):
                t2 = tick()
                print(f"PROF it={it} M={M} data={t_data*1e3:.0f}ms tgt={t_tgt*1e3:.0f}ms "
                      f"model+loss={(t1-t0-t_tgt)*1e3:.0f}ms bwd={(t2-t1)*1e3:.0f}ms", flush=True)
            t_end = time.time()

            if (it + 1) % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                torch.nn.utils.clip_grad_norm_(draft.parameters(), args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                if is_main and gstep % args.log_every == 0:
                    dt = time.time() - t_last
                    t_last = time.time()
                    rec = {"epoch": epoch, "step": gstep, "loss": round(loss.item(), 4),
                           "ce": round(loss_ce.item(), 4), "mse": round(loss_mse_val, 5),
                           "lr": lr_at(gstep), "sec_per_micro": round(dt / (args.log_every * args.accum), 3)}
                    print(json.dumps(rec), flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()

                if args.max_steps and gstep >= args.max_steps:
                    done = True
                    break
        if is_main:
            save_ckpt(epoch)
        if world > 1:
            dist.barrier()
        if done:
            break

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

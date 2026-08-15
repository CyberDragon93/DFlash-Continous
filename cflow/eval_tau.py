"""Measure speculative-decoding acceptance length (tau) for a draft model.

Supports:
  --draft runs/NAME/ckpt.pt            our checkpoints (mask or flow)
  --draft z-lab/Qwen3-4B-DFlash-b16    released DFlash checkpoints (mask)
  --steps K                            flow: K total forwards = (K-1) denoise
                                       ODE/SDE steps + 1 decode forward
  --sde-gamma G                        >0: SDE-style noise re-injection

Greedy verification (temperature 0), DFlash protocol:
tau = mean(accepted_draft_tokens + 1 bonus) per cycle.
"""
import argparse
import glob
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cflow.flow import (
    MODE_DECODE,
    MODE_DENOISE,
    RCF_T_GRIDS,
    FlowConfig,
    endpoint_from_logits,
    inference_t_grid,
    integrator_step,
)
from cflow.modeling import CFlowDraftModel


def load_draft(path_or_id, target, device, args):
    """Returns (draft_model, meta)."""
    if path_or_id.endswith(".pt"):
        ck = torch.load(path_or_id, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        cfg = AutoConfig.from_pretrained(c.get("target", "Qwen/Qwen3-4B"))
        cfg.num_hidden_layers = c["draft_layers"]
        cfg.num_target_layers = target.config.num_hidden_layers
        cfg.block_size = c["block_size"]
        cfg.mask_token_id = c["mask_token_id"]
        cfg.target_layer_ids = c["target_layer_ids"]
        draft = CFlowDraftModel(cfg, mode=c["mode"])
        draft.load_state_dict(ck["model"], strict=True)
        draft = cast_params_bf16(draft, device)
        fcfg = FlowConfig(
            denoise_prob=c.get("denoise_prob", 0.8),
            t_mean=c.get("t_mean", -1.5), t_std=c.get("t_std", 0.8),
            noise_scale=c.get("noise_scale", 2.0),
            decode_noise_scale=c.get("decode_noise_scale", 1.0),
            sde_gamma=args.sde_gamma, grid=args.grid,
        )
        meta = {"mode": c["mode"], "block_size": c["block_size"],
                "mask_token_id": c["mask_token_id"], "fcfg": fcfg,
                "T_r": args.T_r_override or c.get("T_r", 1.0),
                "top_m": c.get("top_m", 64),
                "rcf_harden": args.rcf_harden}
        return draft, meta

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    local = snapshot_download(path_or_id)
    with open(os.path.join(local, "config.json")) as f:
        cj = json.load(f)
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
    for k in ("hidden_size", "intermediate_size", "num_attention_heads",
              "num_key_value_heads", "num_hidden_layers", "num_target_layers",
              "block_size", "head_dim", "rms_norm_eps", "rope_theta"):
        if k in cj:
            setattr(cfg, k, cj[k])
    cfg.mask_token_id = cj["dflash_config"]["mask_token_id"]
    cfg.target_layer_ids = cj["dflash_config"]["target_layer_ids"]
    draft = CFlowDraftModel(cfg, mode="mask")
    sd = {}
    for fp in glob.glob(os.path.join(local, "*.safetensors")):
        sd.update(load_file(fp))
    draft.load_state_dict(sd, strict=True)
    draft = cast_params_bf16(draft, device)
    meta = {"mode": "mask", "block_size": cfg.block_size,
            "mask_token_id": cfg.mask_token_id, "fcfg": FlowConfig()}
    return draft, meta


def cache_repeat(cache, n):
    for l in cache.layers:
        l.batch_repeat_interleave(n)


def cache_select(cache, idx, device):
    sel = torch.tensor([idx], device=device)
    for l in cache.layers:
        l.batch_select_indices(sel)


def cast_params_bf16(draft, device):
    """Cast parameters to bf16 but keep fp buffers (rotary inv_freq,
    emb_mu/sigma) in fp32, matching the training-time numerics."""
    draft = draft.to(device).eval()
    for p in draft.parameters():
        p.data = p.data.to(torch.bfloat16)
    return draft


@torch.inference_mode()
def spec_generate(draft, meta, target, input_ids, max_new_tokens, stop_token_ids,
                  steps=1, collect=None, candidates=1, cand_noise=0.7,
                  cand_mode="noise", cand_temp=0.8):
    device = input_ids.device
    S = meta["block_size"]
    mode = meta["mode"]
    fc = meta["fcfg"]
    n = input_ids.shape[1]
    max_length = n + max_new_tokens
    tl_ids = draft.target_layer_ids
    embed = target.model.embed_tokens
    lm_head_w = target.lm_head.weight
    dtype = lm_head_w.dtype

    out_ids = torch.full((1, max_length + S), meta["mask_token_id"] or 0,
                         dtype=torch.long, device=device)
    out_ids[:, :n] = input_ids

    kv_t = DynamicCache()
    kv_d = DynamicCache()
    pos_all = torch.arange(max_length + S, device=device).unsqueeze(0)

    out = target(input_ids, position_ids=pos_all[:, :n], past_key_values=kv_t,
                 use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out_ids[:, n:n + 1] = out.logits[:, -1:].argmax(-1)
    new_feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)
    new_ctx_pos = pos_all[:, :n]

    start = n
    while start < max_length:
        block_pos = pos_all[:, start:start + S]
        ctx_len = start
        injected = False

        def dstep(inputs, t_val, mode_id):
            nonlocal injected
            Nb = inputs.shape[0]
            nf = None if injected else (new_feats.expand(Nb, -1, -1) if Nb > 1 else new_feats)
            npos = None if injected else (new_ctx_pos.expand(Nb, -1) if Nb > 1 else new_ctx_pos)
            h = draft.infer_step(
                inputs, block_pos.expand(Nb, -1),
                nf, npos, kv_d,
                t=None if t_val is None else torch.full((Nb,), t_val, device=device),
                mode_ids=None if t_val is None else torch.full((Nb,), mode_id, dtype=torch.long, device=device),
            )
            injected = True
            kv_d.crop(ctx_len)
            return h

        if mode == "mask":
            block_ids = out_ids[:, start:start + S].clone()
            block_ids[:, 1:] = meta["mask_token_id"]
            h = dstep(embed(block_ids), None, None)
        elif mode == "rcf":
            D = lm_head_w.shape[1]
            N = candidates
            anchor_emb = embed(out_ids[:, start:start + 1]).view(1, 1, D)
            grid = RCF_T_GRIDS[steps]
            if cand_mode == "sample":
                N = 1  # direct sampling happens in the shared verify section
            x = torch.zeros(N, 1, S, D, device=device, dtype=torch.float32)
            if N > 1 and cand_mode == "noise":
                # candidate 0 stays deterministic (Dirac); others draw from the
                # stochastic source (noise seed = branch selector)
                sigma0 = embed.weight.float().pow(2).mean().sqrt()
                x[1:] = torch.randn(N - 1, 1, S, D, device=device) * (cand_noise * sigma0)
            sc = torch.zeros_like(x)
            h = None
            step0_tokens = None
            sampled_draft = None
            for j, t_j in enumerate(grid):
                inputs = draft.build_rcf_inputs(
                    x.to(dtype), sc.to(dtype),
                    torch.full((N, 1), t_j, device=device), anchor_emb.expand(N, 1, D))
                h = dstep(inputs.view(N, S, D), None, None)
                if j < len(grid) - 1:
                    logits_j = torch.nn.functional.linear(h[:, 1:], lm_head_w)
                    if j == 0:
                        step0_tokens = logits_j.argmax(-1)          # K=1-equivalent draft
                    if N > 1 and cand_mode == "sample-refine" and j == 0:
                        # branch selector = posterior sampling; commit sampled
                        # tokens (hard embed) as the endpoint for candidates >0,
                        # then let the next step re-predict a coherent block
                        probs = torch.softmax(logits_j[1:].float() / cand_temp, dim=-1)
                        samp = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(N - 1, S - 1)
                        x1_hat = endpoint_from_logits(logits_j, embed.weight,
                                                      meta["T_r"], meta["top_m"])
                        x1_hat[1:] = embed(samp).float()
                    elif meta.get("rcf_harden"):
                        # A1 continuity-dial control: hard-commit the argmax
                        # embedding instead of the posterior mixture
                        x1_hat = embed(logits_j.argmax(-1)).float()
                    else:
                        x1_hat = endpoint_from_logits(logits_j, embed.weight,
                                                      meta["T_r"], meta["top_m"])  # [N,15,D]
                    x1_full = torch.zeros_like(x)
                    x1_full[:, 0, 1:, :] = x1_hat
                    x = integrator_step(x, x1_full, t_j, grid[j + 1])
                    sc = x1_full
        else:
            D = lm_head_w.shape[1]
            anchor_emb = embed(out_ids[:, start:start + 1]).view(1, 1, D)
            grid = inference_t_grid(steps, fc)
            # K=1 goes straight to the decode forward: use the decode branch's
            # training-time noise scale, not the denoise-branch scale
            init_scale = fc.noise_scale if steps > 1 else fc.decode_noise_scale
            z = torch.randn(1, 1, S, D, device=device) * init_scale
            x_prev = torch.zeros_like(z)
            for k in range(steps - 1):
                t_k, t_next = grid[k], grid[k + 1]
                z_in, t_in = z, t_k
                if fc.sde_gamma > 0 and k > 0:
                    dt = t_next - t_k
                    alpha = max(1.0 - fc.sde_gamma * dt, 0.0)
                    z_in = alpha * z + (1 - alpha) * torch.randn_like(z) * fc.noise_scale
                    t_in = alpha * t_k
                inputs = draft.build_flow_inputs(z_in.to(dtype), x_prev.to(dtype), anchor_emb, S)
                h = dstep(inputs.view(1, S, D), t_in, MODE_DENOISE)
                x_hat = draft.flow_head(h).float().view(1, 1, S, D)
                z = z + (t_next - t_k) * (x_hat - z) / max(1.0 - t_k, 1e-3)
                x_prev = x_hat
            inputs = draft.build_flow_inputs(z.to(dtype), None, anchor_emb, S)
            h = dstep(inputs.view(1, S, D), 1.0, MODE_DECODE)

        draft_logits = torch.nn.functional.linear(h[:, 1:], lm_head_w)   # [Nc,15,V]
        Nc = draft_logits.shape[0]
        block_out = out_ids[:, start:start + S].clone().repeat(Nc, 1)
        block_out[:, 1:] = draft_logits.argmax(-1)
        if candidates > 1 and cand_mode == "sample" and Nc == 1:
            # best-of-N via posterior sampling: candidate 0 = argmax (floor),
            # candidates 1..N-1 sampled at cand_temp. Works for ALL modes.
            block_out = block_out.repeat(candidates, 1)
            probs = torch.softmax(draft_logits[0].float() / cand_temp, dim=-1)  # [15, V]
            block_out[1:, 1:] = torch.multinomial(probs, candidates - 1, replacement=True).T
            Nc = candidates

        if Nc > 1:
            cache_repeat(kv_t, Nc)
        out = target(block_out, position_ids=block_pos.expand(Nc, -1), past_key_values=kv_t,
                     use_cache=True, output_hidden_states=True)
        posterior = out.logits.argmax(-1)                                # [Nc,S]
        matches = (block_out[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)  # [Nc]
        w = int(matches.argmax())                                        # ties -> candidate 0
        lam = int(matches[w])
        if Nc > 1:
            cache_select(kv_t, w, device)

        bo, po = block_out[w:w + 1], posterior[w:w + 1]
        out_ids[:, start:start + lam + 1] = bo[:, :lam + 1]
        out_ids[:, start + lam + 1] = po[:, lam]
        if collect is not None:
            collect["acc"].append(lam + 1)
            collect["offset_match"].append((bo[0, 1:] == po[0, :-1]).tolist())
            if Nc > 1:
                collect.setdefault("cand0_acc", []).append(int(matches[0]) + 1)
                collect.setdefault("winner_nonzero", 0)
                collect["winner_nonzero"] += int(w != 0)
            if (mode == "rcf" and steps > 1 and step0_tokens is not None
                    and step0_tokens.shape[0] == Nc):
                # flip diagnostics: did later steps change step-0 draft tokens,
                # and did changes move toward or away from the verifier?
                final = bo[0, 1:]
                s0 = step0_tokens[w]
                post = po[0, :-1]
                flip = final != s0
                collect.setdefault("flips", 0)
                collect.setdefault("flip_to_correct", 0)
                collect.setdefault("flip_to_wrong", 0)
                collect.setdefault("positions", 0)
                collect["positions"] += s0.numel()
                collect["flips"] += int(flip.sum())
                collect["flip_to_correct"] += int((flip & (final == post) & (s0 != post)).sum())
                collect["flip_to_wrong"] += int((flip & (final != post) & (s0 == post)).sum())

        feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)
        new_feats = feats[w:w + 1, :lam + 1]
        new_ctx_pos = pos_all[:, start:start + lam + 1]
        start += lam + 1
        kv_t.crop(start)

        # include the bonus token at index `start` (parity with dflash_generate)
        if stop_token_ids is not None and torch.isin(
                out_ids[0, n:start + 1], torch.tensor(stop_token_ids, device=device)).any():
            break

    out_ids = out_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        tail = out_ids[0, n:]
        hits = torch.isin(tail, torch.tensor(stop_token_ids, device=device)).nonzero(as_tuple=True)[0]
        if hits.numel() > 0:
            out_ids = out_ids[:, : n + hits[0] + 1]
    return out_ids


def load_prompts(name, n):
    import datasets
    # templates match dflash/dflash/benchmark.py DATASETS exactly
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    if name == "gsm8k":
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        prompts = [r["question"] + suffix for r in ds]
    elif name == "math500":
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompts = [r["problem"] + suffix for r in ds]
    elif name == "humaneval":
        ds = datasets.load_dataset("openai/openai_humaneval", split="test")
        prompts = ["Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
                   + r["prompt"] + "\n```" for r in ds]
    elif name == "mbpp":
        ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        prompts = [r["prompt"] for r in ds]
    elif name == "mtbench":
        ds = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [(r["prompt"][0] if isinstance(r["prompt"], list) else r["prompt"]) for r in ds]
    else:
        raise ValueError(name)
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--sde-gamma", type=float, default=0.0)
    ap.add_argument("--grid", default="logit-normal", choices=["logit-normal", "uniform"])
    ap.add_argument("--rcf-harden", action="store_true",
                    help="A1 control: hard-commit argmax embedding between steps")
    ap.add_argument("--T-r-override", type=float, default=0.0)
    ap.add_argument("--candidates", type=int, default=1,
                    help="best-of-N drafting (rcf only)")
    ap.add_argument("--cand-noise", type=float, default=0.7)
    ap.add_argument("--cand-mode", default="noise", choices=["noise", "sample", "sample-refine"])
    ap.add_argument("--cand-temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda:0"
    torch.manual_seed(args.seed)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.target)
    draft, meta = load_draft(args.draft, target, device, args)

    prompts = load_prompts(args.dataset, args.n)
    stop_ids = [tok.eos_token_id]

    collect = {"acc": [], "offset_match": []}
    total_tokens = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=False)
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
        out = spec_generate(draft, meta, target, ids, args.max_new, stop_ids,
                            steps=args.steps, collect=collect,
                            candidates=args.candidates, cand_noise=args.cand_noise,
                            cand_mode=args.cand_mode, cand_temp=args.cand_temp)
        total_tokens += out.shape[1] - ids.shape[1]
        if (i + 1) % 20 == 0:
            tau = sum(collect["acc"]) / max(1, len(collect["acc"]))
            print(f"[{i+1}/{len(prompts)}] running tau={tau:.3f}", flush=True)
    wall = time.time() - t0

    acc = collect["acc"]
    tau = sum(acc) / max(1, len(acc))
    S = meta["block_size"]
    offset_rate = []
    for o in range(S - 1):
        reached = [m[o] for m in collect["offset_match"] if len(m) > o and all(m[:o])]
        offset_rate.append(round(sum(reached) / len(reached), 4) if reached else None)
    hist = [acc.count(k) for k in range(1, S + 1)]

    result = {
        "draft": args.draft, "dataset": args.dataset, "n": len(prompts),
        "mode": meta["mode"], "steps": args.steps, "sde_gamma": args.sde_gamma,
        "rcf_harden": args.rcf_harden, "T_r_override": args.T_r_override,
        "candidates": args.candidates, "cand_noise": args.cand_noise,
        "cand_mode": args.cand_mode, "cand_temp": args.cand_temp,
        "cand0_tau": (round(sum(collect["cand0_acc"]) / len(collect["cand0_acc"]), 4)
                      if collect.get("cand0_acc") else None),
        "winner_nonzero_frac": (round(collect.get("winner_nonzero", 0) / max(1, len(acc)), 4)
                                if args.candidates > 1 else None),
        "tau": round(tau, 4), "cycles": len(acc), "total_tokens": total_tokens,
        "tok_per_s": round(total_tokens / wall, 2),
        "acceptance_hist": hist, "offset_match_rate": offset_rate,
        "flip_stats": {k: collect[k] for k in ("positions", "flips", "flip_to_correct", "flip_to_wrong")
                       if k in collect},
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

"""P3: replay prefix-commit probe (master gate for fork/commit training).

For blocks anchored on the target-greedy trajectory, measure per-offset draft
accuracy when offsets 1..c are COMMITTED to the true tokens (clean embeddings)
vs the plain all-mask input (c=0). The uplift U(c) at offsets c+1.. bounds
what fork/commit candidates can gain: a candidate that flips position j to the
true token only extends acceptance if the drafter's continuation AFTER j
adapts to the committed prefix.

Greedy verify makes the final generation exact truth at every depth, so this
is a packed post-hoc replay: 1 target forward + a few packed draft forwards
per prompt. Zero training.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cflow.eval_tau import load_draft, load_prompts, spec_generate


@torch.inference_mode()
def replay(draft, meta, target, seq_ids, prompt_len, depths, device, max_anchors=96):
    """seq_ids: [1, L] target-greedy sequence. Returns per-depth per-offset
    match counts."""
    S = meta["block_size"]
    tl_ids = draft.target_layer_ids
    embed = target.model.embed_tokens
    lm_head_w = target.lm_head.weight
    L = seq_ids.shape[1]

    out = target.model(input_ids=seq_ids, output_hidden_states=True)
    feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)   # [1, L, 5D]

    lo, hi = prompt_len, L - S - 1
    if hi <= lo:
        return None
    anchors = torch.arange(lo, hi, max(1, (hi - lo) // max_anchors), device=device)[:max_anchors]
    A = anchors.shape[0]
    offs = torch.arange(S, device=device)
    pos = anchors.unsqueeze(-1) + offs                                       # [A, S]
    true_tok = seq_ids[0][pos]                                               # [A, S]
    true_emb = embed(true_tok)                                               # [A, S, D]
    mask_emb = embed.weight[meta["mask_token_id"]]

    res = {}
    for c in depths:
        inputs = true_emb.clone()
        inputs[:, c + 1:, :] = mask_emb.to(inputs.dtype)                     # commit 1..c, mask rest
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = draft.forward_blocks(inputs.view(1, A * S, -1).contiguous(),
                                     feats, anchors.unsqueeze(0))
            logits = torch.nn.functional.linear(h, lm_head_w)
        pred = logits.view(A, S, -1).argmax(-1)                              # [A, S]
        match = (pred[:, 1:] == true_tok[:, 1:]).float()                     # [A, 15]
        res[c] = match.mean(0).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--depths", default="0,1,2,4,8")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda:0"
    torch.manual_seed(0)
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    class A:
        sde_gamma = 0.0
        grid = "logit-normal"
        rcf_harden = False
        T_r_override = 0.0
    draft, meta = load_draft(args.draft, target, device, A())
    assert meta["mode"] == "mask", "probe v1 supports mask-mode drafters"

    depths = [int(x) for x in args.depths.split(",")]
    prompts = load_prompts(args.dataset, args.n)
    agg = {c: torch.zeros(15) for c in depths}
    cnt = 0
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=False)
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
        gen = spec_generate(draft, meta, target, ids, args.max_new, [tok.eos_token_id])
        res = replay(draft, meta, target, gen, ids.shape[1], depths, device)
        if res is None:
            continue
        for c in depths:
            agg[c] += torch.tensor(res[c])
        cnt += 1
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(prompts)}]", flush=True)

    table = {}
    for c in depths:
        table[c] = [round(x, 4) for x in (agg[c] / cnt).tolist()]
        print(f"commit c={c:2d}: " + " ".join(f"{x:.3f}" for x in table[c]))
    # headline: accuracy at the first free offset after the commit
    print("\nU(c) = acc at offset c+1 | baseline (c=0) at same offset:")
    for c in depths:
        if c == 0 or c + 1 > 15:
            continue
        print(f"  c={c}: {table[c][c]:.4f} vs {table[0][c]:.4f}  (+{table[c][c]-table[0][c]:.4f})")
    if args.out:
        json.dump({"draft": args.draft, "dataset": args.dataset, "n": cnt,
                   "per_offset_match_by_depth": table}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

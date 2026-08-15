"""A/B parity check: official dflash generate vs our eval harness on the same
prompts with the released z-lab checkpoint. Prints tau from both."""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "dflash")  # repo root containing the dflash package
from dflash.model import DFlashDraftModel, dflash_generate  # noqa: E402

from cflow.eval_tau import load_draft, load_prompts, spec_generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    device = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    theirs = DFlashDraftModel.from_pretrained("z-lab/Qwen3-4B-DFlash-b16", dtype=torch.bfloat16).to(device).eval()

    class Args:
        sde_gamma = 0.0
        grid = "logit-normal"
    ours, meta = load_draft("z-lab/Qwen3-4B-DFlash-b16", target, device, Args())

    prompts = load_prompts("gsm8k", args.n)
    stop = [tok.eos_token_id]

    acc_t, acc_o = [], []
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=False)
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)

        r = dflash_generate(theirs, target, ids, max_new_tokens=args.max_new,
                            stop_token_ids=stop, temperature=0.0, return_stats=True)
        acc_t.extend(r.acceptance_lengths)

        collect = {"acc": [], "offset_match": []}
        out = spec_generate(ours, meta, target, ids, args.max_new, stop, steps=1, collect=collect)
        acc_o.extend(collect["acc"])

        print(f"[{i}] theirs cycles={len(r.acceptance_lengths)} "
              f"mean={sum(r.acceptance_lengths)/len(r.acceptance_lengths):.3f} | "
              f"ours cycles={len(collect['acc'])} "
              f"mean={sum(collect['acc'])/len(collect['acc']):.3f} | "
              f"theirs_tokens={r.num_output_tokens} ours_tokens={out.shape[1]-ids.shape[1]}",
              flush=True)

    print(f"\nTOTAL theirs tau={sum(acc_t)/len(acc_t):.4f} ({len(acc_t)} cycles) | "
          f"ours tau={sum(acc_o)/len(acc_o):.4f} ({len(acc_o)} cycles)")


if __name__ == "__main__":
    main()

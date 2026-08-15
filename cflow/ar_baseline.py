"""Measure plain autoregressive (greedy, no speculation) tok/s for the target
model under the same harness conditions (bs=1, HF eager, bf16)."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cflow.eval_tau import load_prompts


def main():
    device = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    for ds, n in (("gsm8k", 20), ("mtbench", 10)):
        prompts = load_prompts(ds, n)
        total_tokens, t0 = 0, time.time()
        for p in prompts:
            enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                          add_generation_prompt=True, enable_thinking=False)
            ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
            with torch.inference_mode():
                out = target.generate(ids, max_new_tokens=1024, do_sample=False,
                                      eos_token_id=tok.eos_token_id,
                                      pad_token_id=tok.eos_token_id)
            total_tokens += out.shape[1] - ids.shape[1]
        wall = time.time() - t0
        print(f"AR {ds}: {total_tokens} tokens in {wall:.1f}s -> {total_tokens/wall:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()

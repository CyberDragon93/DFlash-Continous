"""Demo: does the continuous-flow drafter actually generate sentences?

Mode A (spec): normal speculative decoding (verified output + tau).
Mode B (free-run): the draft generates EVERYTHING — all 15 block tokens are
accepted without verification and the next anchor is the draft's own token at
offset 15. The target model only supplies hidden features for the committed
text (its logits are never consulted). 100% flow-generated text.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cflow.eval_tau import load_draft, spec_generate
from cflow.flow import RCF_T_GRIDS, endpoint_from_logits, integrator_step

PROMPTS = [
    "Janet has 3 apples and buys 2 more bags with 6 apples each. How many apples does she have now?",
    "Explain in two or three sentences what speculative decoding is and why it speeds up LLM inference.",
    "用两三句话介绍一下什么是块扩散语言模型。",
]


@torch.inference_mode()
def free_run(draft, meta, target, input_ids, max_new, steps=2):
    device = input_ids.device
    S = meta["block_size"]
    tl_ids = draft.target_layer_ids
    embed = target.model.embed_tokens
    lm_head_w = target.lm_head.weight
    dtype = lm_head_w.dtype
    D = lm_head_w.shape[1]
    n = input_ids.shape[1]
    max_length = n + max_new

    out_ids = torch.zeros(1, max_length + S, dtype=torch.long, device=device)
    out_ids[:, :n] = input_ids
    kv_t, kv_d = DynamicCache(), DynamicCache()
    pos_all = torch.arange(max_length + S, device=device).unsqueeze(0)

    # prefill: target supplies features only; FIRST token also from draft?
    # the drafter needs an anchor; take the last prompt token as anchor by
    # drafting the block starting AT position n with anchor = prompt's last
    # predicted continuation. To stay 100% draft-generated after the prompt,
    # we anchor block 1 on the target's prefill argmax ONCE (unavoidable seed),
    # then never consult target logits again.
    out = target(input_ids, position_ids=pos_all[:, :n], past_key_values=kv_t,
                 use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out_ids[:, n:n + 1] = out.logits[:, -1:].argmax(-1)
    new_feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)
    new_ctx_pos = pos_all[:, :n]

    start = n
    while start < max_length:
        block_pos = pos_all[:, start:start + S]
        ctx_len = start
        anchor_emb = embed(out_ids[:, start:start + 1]).view(1, 1, D)
        grid = RCF_T_GRIDS[steps]
        x = torch.zeros(1, 1, S, D, device=device)
        sc = torch.zeros_like(x)
        h = None
        injected = False
        for j, t_j in enumerate(grid):
            inputs = draft.build_rcf_inputs(x.to(dtype), sc.to(dtype),
                                            torch.full((1, 1), t_j, device=device), anchor_emb)
            h = draft.infer_step(inputs.view(1, S, D), block_pos,
                                 None if injected else new_feats,
                                 None if injected else new_ctx_pos, kv_d)
            injected = True
            kv_d.crop(ctx_len)
            if j < len(grid) - 1:
                lj = torch.nn.functional.linear(h[:, 1:], lm_head_w)
                x1 = endpoint_from_logits(lj, embed.weight, meta["T_r"], meta["top_m"])
                xf = torch.zeros_like(x)
                xf[:, 0, 1:, :] = x1
                x = integrator_step(x, xf, t_j, grid[j + 1])
                sc = xf

        drafted = torch.nn.functional.linear(h[:, 1:], lm_head_w).argmax(-1)  # [1,15]
        block = out_ids[:, start:start + S].clone()
        block[:, 1:] = drafted
        # commit offsets 0..14 as text; offset 15 becomes the next anchor
        out_ids[:, start:start + S] = block
        # target forward ONLY to refresh features for committed positions
        out = target(block, position_ids=block_pos, past_key_values=kv_t,
                     use_cache=True, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in tl_ids], dim=-1)
        new_feats = feats[:, :S - 1]
        new_ctx_pos = pos_all[:, start:start + S - 1]
        start += S - 1
        kv_t.crop(start)

    return out_ids[:, :max_length]


def main():
    device = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    class A:
        sde_gamma = 0.0
        grid = "logit-normal"
        rcf_harden = False
        T_r_override = 0.0
    draft, meta = load_draft("runs/rcf_v2/ckpt.pt", target, device, A())

    for p in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=False)
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)

        collect = {"acc": [], "offset_match": []}
        o = spec_generate(draft, meta, target, ids, 160, [tok.eos_token_id],
                          steps=2, collect=collect)
        tau = sum(collect["acc"]) / max(1, len(collect["acc"]))
        print("=" * 90)
        print("PROMPT:", p)
        print(f"\n--- A) 投机解码输出 (target 验证, tau={tau:.2f}) ---")
        print(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))

        o2 = free_run(draft, meta, target, ids, 160, steps=2)
        print("\n--- B) FLOW DRAFTER 自由生成 (无验证, 100% draft tokens) ---")
        print(tok.decode(o2[0, ids.shape[1]:], skip_special_tokens=True))
        print()


if __name__ == "__main__":
    main()

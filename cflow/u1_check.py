"""U1 parity: warm-started UNTRAINED rcf draft at K=1 must produce identical
outputs and acceptance sequences to the released mask checkpoint."""
import glob
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cflow.eval_tau import load_draft, load_prompts, spec_generate
from cflow.flow import FlowConfig
from cflow.modeling import CFlowDraftModel


def main():
    device = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    class A:
        sde_gamma = 0.0
        grid = "logit-normal"
    mask_draft, mask_meta = load_draft("z-lab/Qwen3-4B-DFlash-b16", target, device, A())

    local = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    cj = json.load(open(os.path.join(local, "config.json")))
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
    cfg.num_hidden_layers = cj["num_hidden_layers"]
    cfg.num_target_layers = cj["num_target_layers"]
    cfg.block_size = cj["block_size"]
    cfg.mask_token_id = cj["dflash_config"]["mask_token_id"]
    cfg.target_layer_ids = cj["dflash_config"]["target_layer_ids"]
    rcf = CFlowDraftModel(cfg, mode="rcf")
    sd = {}
    for fp in glob.glob(os.path.join(local, "*.safetensors")):
        sd.update(load_file(fp))
    missing, unexpected = rcf.load_state_dict(sd, strict=False)
    assert not unexpected
    rcf.init_rcf_bias(target.model.embed_tokens.weight[cfg.mask_token_id].detach().float())
    rcf = rcf.to(device=device, dtype=torch.bfloat16).eval()
    rcf_meta = {"mode": "rcf", "block_size": cfg.block_size,
                "mask_token_id": cfg.mask_token_id, "fcfg": FlowConfig(),
                "T_r": 1.0, "top_m": 64}

    prompts = load_prompts("gsm8k", 3)
    stop = [tok.eos_token_id]
    ok = True
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=False)
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
        c1 = {"acc": [], "offset_match": []}
        o1 = spec_generate(mask_draft, mask_meta, target, ids, 256, stop, steps=1, collect=c1)
        c2 = {"acc": [], "offset_match": []}
        o2 = spec_generate(rcf, rcf_meta, target, ids, 256, stop, steps=2, collect=c2)
        c3 = {"acc": [], "offset_match": []}
        o3 = spec_generate(rcf, rcf_meta, target, ids, 256, stop, steps=1, collect=c3)
        same = torch.equal(o1, o3) and c1["acc"] == c3["acc"]
        ok &= same
        print(f"[{i}] mask acc={c1['acc'][:8]}... | rcf K=1 acc={c3['acc'][:8]}... "
              f"IDENTICAL={same} | rcf K=2 (untrained) tau={sum(c2['acc'])/len(c2['acc']):.3f} "
              f"vs K=1 tau={sum(c3['acc'])/len(c3['acc']):.3f}")
    print("U1 PARITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

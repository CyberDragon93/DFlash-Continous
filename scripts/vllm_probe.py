from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=2048,
          gpu_memory_utilization=0.85, attention_backend="FLASH_ATTN")
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8))
print("GEN OK:", out[0].outputs[0].text)

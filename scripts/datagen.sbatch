#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --requeue
#SBATCH -J cflow_datagen
#SBATCH -o ${PROJECT_DIR:?set PROJECT_DIR}/logs/%x_%j.out

set -uo pipefail
BASE=${PROJECT_DIR:?set PROJECT_DIR}
PY=${VLLM_ENV:?set VLLM_ENV}/bin/python
export HF_HOME=${HF_CACHE:-$HOME/.cache/huggingface}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
export TRITON_CACHE_DIR=/tmp/triton
# compute nodes have no nvcc; avoid ALL FlashInfer JIT paths
# (attention backend is set via LLM(attention_backend="FLASH_ATTN") in the script)
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$BASE/data/gen"

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i "$PY" "$BASE/cflow/data/generate_responses.py" \
    --prompts "$BASE/data/prompts.jsonl" \
    --out "$BASE/data/gen/shard_$i.jsonl" \
    --shard "$i" --num-shards 4 \
    > "$BASE/logs/datagen_shard_$i.log" 2>&1 &
done
wait

n_done=$(ls "$BASE"/data/gen/shard_*.jsonl 2>/dev/null | wc -l)
if [ "$n_done" -eq 4 ]; then
  cat "$BASE"/data/gen/shard_*.jsonl > "$BASE/data/train_raw.jsonl.tmp"
  mv "$BASE/data/train_raw.jsonl.tmp" "$BASE/data/train_raw.jsonl"
  wc -l "$BASE/data/train_raw.jsonl"
else
  echo "only $n_done/4 shards complete; requeue will resume"
  exit 1
fi

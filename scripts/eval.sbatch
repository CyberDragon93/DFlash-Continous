#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --requeue
#SBATCH -J cflow_eval
#SBATCH -o logs/%x_%j.out

# Usage: sbatch --export=ALL,PLAN=eval_plans/foo.txt scripts/eval.sbatch
# PLAN file: one cflow.eval_tau arg-string per line, e.g.
#   --draft runs/flow_v1/ckpt.pt --dataset gsm8k --steps 2 --n 200 --out runs/flow_v1/gsm8k_k2.json
set -uo pipefail
BASE=${PROJECT_DIR:?set PROJECT_DIR}
source ${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh
conda activate dl
export HF_HOME=${HF_CACHE:-$HOME/.cache/huggingface}
cd "$BASE"

PLAN=${PLAN:?set PLAN}
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  out=$(echo "$line" | grep -oE -- "--out [^ ]+" | cut -d" " -f2)
  if [ -n "$out" ] && [ -f "$out" ]; then
    echo "skip (done): $out"
    continue
  fi
  gpu=$((i % 4))
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python -m cflow.eval_tau $line \
    > "logs/eval_${SLURM_JOB_ID}_${i}.log" 2>&1 &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done < "$PLAN"
wait
echo "all $i evals done"
grep -h '"tau"' logs/eval_${SLURM_JOB_ID}_*.log || true

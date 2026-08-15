#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --wckey=YOUR_WCKEY
#SBATCH --requeue
#SBATCH -J cflow_train
#SBATCH -o ${PROJECT_DIR:?set PROJECT_DIR}/logs/%x_%j.out

# Usage: sbatch --export=ALL,NAME=flow_v1,MODE=flow,EXTRA="--epochs 6" [-J name] scripts/train.sbatch
set -uo pipefail
BASE=${PROJECT_DIR:?set PROJECT_DIR}
source ${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh
conda activate dl
export HF_HOME=${HF_CACHE:-$HOME/.cache/huggingface}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
export TRITON_CACHE_DIR=/tmp/triton
export OMP_NUM_THREADS=8
cd "$BASE"

NAME=${NAME:?set NAME}
MODE=${MODE:?set MODE}
DATA=${DATA:-data/train_raw.jsonl}
EXTRA=${EXTRA:-}

RESUME=""
if [ -f "runs/$NAME/ckpt.pt" ]; then
  RESUME="--resume runs/$NAME/ckpt.pt"
fi

torchrun --nproc_per_node=4 -m cflow.train \
  --data "$DATA" --mode "$MODE" --name "$NAME" $RESUME $EXTRA

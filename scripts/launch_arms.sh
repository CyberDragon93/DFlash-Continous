#!/bin/bash
# Launch the main comparison arms.
set -euo pipefail
BASE=$PROJECT_DIR
cd "$BASE"
test -f data/train_raw.jsonl || { echo "train_raw.jsonl missing"; exit 1; }

# Arm 1: mask baseline from scratch (paper-ablation-style), 6 epochs
sbatch -J cflow_mask_scratch \
  --export=ALL,NAME=mask_scratch,MODE=mask,EXTRA="--epochs 6 --lr 6e-4" scripts/train.sbatch

# Arm 2: mask warm-start continue-train (control for RCF's warm start), 4 epochs
sbatch -J cflow_mask_ws \
  --export=ALL,NAME=mask_ws,MODE=mask,EXTRA="--warm-start zlab --epochs 4 --lr 2e-4 --weight-decay 0.01 --beta2 0.95" \
  scripts/train.sbatch

# Arm 3: RCF v1 (rollout-consistent continuous flow), warm-start, 4 epochs
sbatch -J cflow_rcf_v1 \
  --export=ALL,NAME=rcf_v1,MODE=rcf,EXTRA="--warm-start zlab --epochs 4 --lr 2e-4 --weight-decay 0.01 --beta2 0.95 --p-roll 0.5 --depth2-prob 0.15" \
  scripts/train.sbatch

# Arm 4: ELF v1 (embedding flow matching), scratch, 6 epochs
sbatch -J cflow_elf_v1 \
  --export=ALL,NAME=elf_v1,MODE=flow,EXTRA="--epochs 6 --lr 6e-4" scripts/train.sbatch

squeue -u "$USER" -o "%A %j %T %R" | grep cflow || true

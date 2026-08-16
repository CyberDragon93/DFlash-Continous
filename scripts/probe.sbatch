#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p batch
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --wckey=YOUR_WCKEY
#SBATCH -J cflow_probe
#SBATCH -o logs/%x_%j.out

source ${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh
conda activate dl
export HF_HOME=${HF_CACHE:-$HOME/.cache/huggingface}
cd ${PROJECT_DIR:?set PROJECT_DIR}
CUDA_VISIBLE_DEVICES=0 python -m cflow.probe_commit --draft runs/mask_ws/ckpt.pt --dataset gsm8k --n 100 --out runs/mask_ws/probe_commit.json

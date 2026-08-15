#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --wckey=YOUR_WCKEY
#SBATCH -J cflow_demo
#SBATCH -o ${PROJECT_DIR:?set PROJECT_DIR}/logs/%x_%j.out

source ${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh
conda activate dl
export HF_HOME=${HF_CACHE:-$HOME/.cache/huggingface}
cd ${PROJECT_DIR:?set PROJECT_DIR}
CUDA_VISIBLE_DEVICES=0 python -m cflow.demo_generate

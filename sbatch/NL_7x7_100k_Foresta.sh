#!/usr/bin/env bash
#SBATCH --job-name=nl_7x7_100k_dynamic
#SBATCH --chdir=/home/neerattr/numberlinkSolver
#SBATCH --output=/home/neerattr/numberlinkSolver/logs/%j.out
#SBATCH --error=/home/neerattr/numberlinkSolver/logs/%j.err
#SBATCH --partition=gpu-A100
#SBATCH --account=agostinelli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -euo pipefail

module load python3/anaconda/3.12
source /home/neerattr/.venv/bin/activate

cd /home/neerattr/numberlinkSolver

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

python - << 'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("CUDA OK, device_count=", torch.cuda.device_count())
PY

python train_numberlink.py \
  --domain numberlink.7x7x3_random_walk \
  --heur resnet_fc.1024H_2B_bn \
  --max_itrs 100000 \
  --search_itrs 200 \
  --up_itrs 800 \
  --batch_size 2048 \
  --step_max 1000 \
  --procs "${SLURM_CPUS_PER_TASK}" \
  --curriculum \
  --out_dir "/home/neerattr/numberlinkSolver/runs/${SLURM_JOB_ID}"

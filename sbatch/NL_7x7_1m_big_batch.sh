#!/usr/bin/env bash
#SBATCH --job-name=nl_7x7_1m
#SBATCH --chdir=/home/neerattr/numberlinkSolver
#SBATCH --output=/home/neerattr/numberlinkSolver/logs/%j.out
#SBATCH --error=/home/neerattr/numberlinkSolver/logs/%j.err
#SBATCH --partition=gpu-A100
#SBATCH --account=agostinelli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
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
  --heur resnet_fc.256H_2B_bn \
  --max_itrs 1000000 \
  --search_itrs 400 \
  --batch_size 512 \
  --up_batch_size 32 \
  --up_nnet_batch_size 4096 \
  --step_max 10 \
  --procs "${SLURM_CPUS_PER_TASK}" \
  --debug \
  --curriculum \
  --out_dir "runs/${SLURM_JOB_ID}"

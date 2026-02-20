#!/usr/bin/env bash
#SBATCH --job-name=nl_eval_319399_s10
#SBATCH --chdir=/home/neerattr/numberlinkSolver
#SBATCH --output=/home/neerattr/numberlinkSolver/logs/%j.out
#SBATCH --error=/home/neerattr/numberlinkSolver/logs/%j.err
#SBATCH --partition=gpu-A100
#SBATCH --account=agostinelli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

module load python3/anaconda/3.12
source /home/neerattr/.venv/bin/activate

cd /home/neerattr/numberlinkSolver

RUN_ID=319399
EVAL_FILE="/home/neerattr/numberlinkSolver/eval/7x7x3_s10_2k.pkl"
RESULTS_DIR="/home/neerattr/numberlinkSolver/eval_results/${RUN_ID}_s10_bwas"
HEUR_FILE="/home/neerattr/numberlinkSolver/runs/${RUN_ID}/heur_targ.pt"

mkdir -p /home/neerattr/numberlinkSolver/eval /home/neerattr/numberlinkSolver/eval_results

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

if [[ ! -f "${HEUR_FILE}" ]]; then
  echo "Missing heuristic file: ${HEUR_FILE}"
  exit 1
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
  echo "Generating held-out eval set: ${EVAL_FILE}"
  python -m deepxube problem_inst \
    --domain numberlink.7x7x3_random_walk \
    --step_max 10 \
    --num 2000 \
    --file "${EVAL_FILE}"
else
  echo "Using existing eval set: ${EVAL_FILE}"
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
  echo "Eval generation did not produce file: ${EVAL_FILE}"
  exit 1
fi

echo "Running solve with heuristic ${HEUR_FILE}"
python -m deepxube solve \
  --domain numberlink.7x7x3_random_walk \
  --heur resnet_fc.256H_2B_bn \
  --heur_type V \
  --heur_file "${HEUR_FILE}" \
  --pathfind bwas.1_1.0_0.0 \
  --file "${EVAL_FILE}" \
  --results "${RESULTS_DIR}" \
  --redo

echo "Summarizing results"
if [[ -f "${RESULTS_DIR}/results.pkl" ]]; then
  python scripts/summarize_solve_results.py --results "${RESULTS_DIR}/results.pkl"
else
  echo "Missing solve output: ${RESULTS_DIR}/results.pkl"
  exit 1
fi

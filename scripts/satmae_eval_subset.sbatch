#!/bin/bash
#SBATCH --job-name=satmae-eval-subset
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/satmae_eval_subset_%j.out
#SBATCH --error=logs/satmae_eval_subset_%j.err

set -euo pipefail

# Submit from the foundry repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs

# Data, weights, and results live outside the foundry checkout.
export SATMAE_WORKDIR="${SATMAE_WORKDIR:-$HOME/scratch/satmae}"

echo "== job: eval_subset  job_id: ${SLURM_JOB_ID}  node: $(hostname) =="
date

echo "== nvidia-smi ==" | tee "logs/satmae_eval_subset_${SLURM_JOB_ID}_nvidia-smi.txt"
nvidia-smi | tee -a "logs/satmae_eval_subset_${SLURM_JOB_ID}_nvidia-smi.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate ~/scratch/envs/satmae; set -u

echo "== torch/CUDA versions ==" | tee "logs/satmae_eval_subset_${SLURM_JOB_ID}_versions.txt"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())" \
  | tee -a "logs/satmae_eval_subset_${SLURM_JOB_ID}_versions.txt"

START_TIME=$(date +%s)

python scripts/satmae_evaluate.py \
  --csv "${SATMAE_WORKDIR}/data/val_subset5000.csv" \
  --data-root "${SATMAE_WORKDIR}/data" \
  --checkpoint "${SATMAE_WORKDIR}/weights/finetune-vit-base-e7.pth" \
  --batch-size 64 \
  --num-workers 8 \
  --out-dir "${SATMAE_WORKDIR}/results/subset5000"

END_TIME=$(date +%s)
echo "Wall time (s): $((END_TIME - START_TIME))" | tee "logs/satmae_eval_subset_${SLURM_JOB_ID}_walltime.txt"

echo "== sacct =="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize \
  | tee "logs/satmae_eval_subset_${SLURM_JOB_ID}_sacct.txt"

date
echo "== done =="

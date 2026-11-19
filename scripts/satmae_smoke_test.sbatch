#!/bin/bash
#SBATCH --job-name=satmae-smoke
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/satmae_smoke_%j.out
#SBATCH --error=logs/satmae_smoke_%j.err

set -euo pipefail

# Submit from the foundry repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs

# Data, weights, and results live outside the foundry checkout.
export SATMAE_WORKDIR="${SATMAE_WORKDIR:-$HOME/scratch/satmae}"

echo "== job: smoke_test  job_id: ${SLURM_JOB_ID}  node: $(hostname) =="
date

echo "== nvidia-smi ==" | tee "logs/satmae_smoke_${SLURM_JOB_ID}_nvidia-smi.txt"
nvidia-smi | tee -a "logs/satmae_smoke_${SLURM_JOB_ID}_nvidia-smi.txt"

# Activate environment -- adjust to how conda is set up on the cluster.
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate ~/scratch/envs/satmae; set -u

echo "== torch/CUDA versions ==" | tee "logs/satmae_smoke_${SLURM_JOB_ID}_versions.txt"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())" \
  | tee -a "logs/satmae_smoke_${SLURM_JOB_ID}_versions.txt"

START_TIME=$(date +%s)

python scripts/satmae_evaluate.py \
  --csv "${SATMAE_WORKDIR}/data/val_prepared.csv" \
  --data-root "${SATMAE_WORKDIR}/data" \
  --checkpoint "${SATMAE_WORKDIR}/weights/finetune-vit-base-e7.pth" \
  --limit 100 \
  --batch-size 32 \
  --num-workers 4 \
  --out-dir "${SATMAE_WORKDIR}/results/smoke"

END_TIME=$(date +%s)
echo "Wall time (s): $((END_TIME - START_TIME))" | tee "logs/satmae_smoke_${SLURM_JOB_ID}_walltime.txt"

echo "== sacct =="
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize \
  | tee "logs/satmae_smoke_${SLURM_JOB_ID}_sacct.txt"

date
echo "== done =="

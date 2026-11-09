#!/bin/bash
#SBATCH --job-name=satmae-temporal-random
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/satmae_temporal_random_%j.out
#SBATCH --error=logs/satmae_temporal_random_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=satmae_temporal_gpu_common.sh
source "$REPO_ROOT/scripts/satmae_temporal_gpu_common.sh" random

OUT_DIR="$SATMAE_TEMPORAL_WORKDIR/results/random"
mkdir -p "$OUT_DIR"

START=$(date +%s)
python "$REPO_ROOT/scripts/satmae_temporal_evaluate.py" \
  --csv "$SATMAE_TEMPORAL_WORKDIR/data/val_prepared.csv" \
  --eval-csv "$SATMAE_TEMPORAL_WORKDIR/data/val_subset500.csv" \
  --code-dir "$SATMAE_CODE_DIR" \
  --out-dir "$OUT_DIR" \
  --batch-size 8 \
  --num-workers 4 \
  --random-init \
  --seed 0
END=$(date +%s)
echo "wall_seconds=$((END-START))" | tee "logs/random_${SLURM_JOB_ID}_walltime.txt"

sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode \
  > "logs/random_${SLURM_JOB_ID}_sacct.txt" || true

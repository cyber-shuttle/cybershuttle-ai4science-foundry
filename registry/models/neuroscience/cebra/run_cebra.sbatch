#!/bin/bash
#SBATCH --job-name=cebra-hippocampus
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=ece 
#SBATCH --partition=ice-gpu


set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source common.sh

ITERS=${ITERS:-5000}
OUT=outputs/run_${SLURM_JOB_ID}
python demo_hippocampus.py --out-dir "$OUT" --max-iterations "$ITERS"
set +e
python validate.py "$OUT" | tee "$OUT/validation.txt"
VAL_STATUS=${PIPESTATUS[0]}
set -e

finish_job "$OUT"
echo "validation_exit_code=$VAL_STATUS (0 = all checks passed)"
exit "$VAL_STATUS"

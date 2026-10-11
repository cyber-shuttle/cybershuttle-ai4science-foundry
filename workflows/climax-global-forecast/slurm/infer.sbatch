#!/bin/bash
#SBATCH --job-name=climax-inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/climax-inference-%j.out
#SBATCH --error=logs/climax-inference-%j.err

set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_DIR"

: "${CLIMAX_SOURCE:?Set CLIMAX_SOURCE to the pinned ClimaX source checkout}"
: "${CHECKPOINT:?Set CHECKPOINT to the downloaded 5.625deg.ckpt}"
: "${ERA5_NPZ:?Set ERA5_NPZ to the prepared ERA5 npz directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the run output directory}"
: "${CLIMAX_PYTHON:?Set CLIMAX_PYTHON to the Python executable in the Conda environment}"

export CLIMAX_COMMIT
CLIMAX_COMMIT="$(git -C "$CLIMAX_SOURCE" rev-parse HEAD)"
mkdir -p "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "node=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "climax_commit=$CLIMAX_COMMIT"
echo "python=$CLIMAX_PYTHON"
echo "checkpoint=$CHECKPOINT"
echo "data=$ERA5_NPZ"
nvidia-smi || true

"$CLIMAX_PYTHON" scripts/infer.py \
  --climax-source "$CLIMAX_SOURCE" \
  --config "$CLIMAX_SOURCE/configs/global_forecast_climax.yaml" \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$ERA5_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --time-index 0 \
  --lead-hours 6 \
  --device cuda

echo "end=$(date --iso-8601=seconds)"
nvidia-smi || true

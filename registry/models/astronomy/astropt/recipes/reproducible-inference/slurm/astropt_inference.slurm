#!/bin/bash
#SBATCH --job-name=astropt-a2
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/astropt_a2_%j.out
#SBATCH --error=logs/astropt_a2_%j.err

set -euo pipefail

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export HF_HOME="$PROJ/hf_cache"
ASTROPT_SOURCE_COMMIT="fa361ae5b6631c89525429755ed3fc2b2d304eca"

conda activate "$PROJ/env"
cd "$PROJ"

echo "========================================"
echo "AstroPT Assignment 2 Reproducible Run"
echo "========================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== Software ==="
python --version
python - <<'PY'
import torch
import importlib.metadata

print("AstroPT:", importlib.metadata.version("astropt"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo
echo "=== AstroPT Upstream Reference Commit ==="
echo "$ASTROPT_SOURCE_COMMIT"

echo
echo "=== Smoke Test ==="
python scripts/smoke_load.py

echo
echo "=== Small Inference ==="
python scripts/inference_small.py

echo
echo "=== Scientific Validation ==="
python scripts/validate_mag_g.py

echo
echo "========================================"
echo "WORKFLOW COMPLETED SUCCESSFULLY"
echo "Finished: $(date)"
echo "========================================"

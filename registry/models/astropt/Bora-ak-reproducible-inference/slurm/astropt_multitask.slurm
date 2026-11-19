#!/bin/bash
#SBATCH --job-name=astropt-a5
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/astropt_a5_%j.out
#SBATCH --error=logs/astropt_a5_%j.err

set -euo pipefail

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export HF_HOME="$PROJ/hf_cache"

conda activate "$PROJ/env"
cd "$PROJ"

echo "========================================"
echo "AstroPT Assignment 5 Multi-Task Probe"
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
import importlib.metadata
import torch
print("AstroPT:", importlib.metadata.version("astropt"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
echo
echo "=== Multi-Task Validation ==="
python scripts/validate_multitask.py --num-examples "${NUM_EXAMPLES:-3000}"
echo
echo "========================================"
echo "ASSIGNMENT 5 WORKFLOW COMPLETED SUCCESSFULLY"
echo "Finished: $(date)"
echo "========================================"

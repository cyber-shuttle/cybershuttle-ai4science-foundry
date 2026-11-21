#!/bin/bash
#SBATCH --job-name=cebra-robustness
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
module load python/3.12.5
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/robustness

VENV_PATH="${CEBRA_VENV:-${SLURM_SUBMIT_DIR}/../cebra-venv}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: CEBRA virtual environment was not found at ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

nvidia-smi -L
python -V
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python scripts/robustness.py

#!/bin/bash
# One-time environment setup for the HyenaDNA inference workflow.
# Run on a login node:  bash setup_env.sh
# Override locations:   SCRATCH_DIR=/path ENV_PREFIX=/path/to/env bash setup_env.sh

# No -u: conda's own scripts reference unset variables (see README troubleshooting).
set -eo pipefail
cd "$(dirname "$0")"

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH_DIR/.conda/envs/hyenadna_env}"

if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: scratch directory '$SCRATCH_DIR' does not exist."
    echo "On PACE ICE it is normally ~/scratch. On another system, run:"
    echo "  SCRATCH_DIR=/path/to/your/scratch bash setup_env.sh"
    exit 1
fi

# Keep caches off the quota-limited home directory
export CONDA_PKGS_DIRS="$SCRATCH_DIR/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_DIR/.pip_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "Env already exists at $ENV_PREFIX, reusing it"
else
    conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX"

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Installed versions ==="
python --version
python -c "import torch, transformers, numpy, huggingface_hub; print('torch', torch.__version__); print('transformers', transformers.__version__); print('numpy', numpy.__version__); print('huggingface_hub', huggingface_hub.__version__)"
echo "Env ready at: $ENV_PREFIX"
echo "CUDA availability is checked inside the GPU job, since login nodes have no GPU."
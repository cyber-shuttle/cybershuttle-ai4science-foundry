#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH_ROOT/.conda/envs/climax-inference}"

if ! command -v module >/dev/null 2>&1; then
  echo "Environment modules are unavailable. Run this on a PACE ICE login/compute shell." >&2
  exit 1
fi

module load miniforge
if ! command -v conda >/dev/null 2>&1; then
  echo "The miniforge module loaded but conda is still unavailable." >&2
  echo "Check PACE's current module instructions (for example: module avail miniforge)." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

export CONDA_PKGS_DIRS="$SCRATCH_ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/.cache/pip"
export TMPDIR="$SCRATCH_ROOT/tmp"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export HF_HOME="$SCRATCH_ROOT/.cache/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/.cache/torch"
export GIT_CONFIG_GLOBAL="$SCRATCH_ROOT/.gitconfig"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
mkdir -p "$(dirname "$ENV_PREFIX")"

if [ -f "$ENV_PREFIX/conda-meta/history" ]; then
  echo "Using existing environment at $ENV_PREFIX"
else
  echo "Creating ClimaX environment at $ENV_PREFIX"
  conda env create --prefix "$ENV_PREFIX" --file "$WORKFLOW_DIR/environment.yml"
fi

conda activate "$ENV_PREFIX"
echo "Environment: $CONDA_PREFIX"
python --version
python -c 'import importlib.metadata as metadata; import jsonargparse, rich, tensorboard, timm, torch, torchdata, torchvision, pytorch_lightning; from pytorch_lightning.callbacks import RichProgressBar; from pytorch_lightning.loggers import TensorBoardLogger; print("PyTorch:", torch.__version__); print("torchvision:", metadata.version("torchvision")); print("PyTorch Lightning:", metadata.version("pytorch-lightning")); print("timm:", metadata.version("timm")); print("torchdata:", metadata.version("torchdata")); print("jsonargparse:", metadata.version("jsonargparse")); print("rich:", metadata.version("rich")); print("TensorBoard:", metadata.version("tensorboard")); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
conda list --explicit > "$WORKFLOW_DIR/execution/conda-explicit.txt"
pip freeze > "$WORKFLOW_DIR/execution/pip-freeze.txt"
if ! python -c 'import torch; x = torch.ones(1, device="cuda"); torch.cuda.synchronize(); print("CUDA device:", torch.cuda.get_device_name(0)); print("CUDA capability:", torch.cuda.get_device_capability(0))'; then
  echo "CUDA smoke check failed; do not submit ClimaX GPU jobs from this environment." >&2
  exit 1
fi

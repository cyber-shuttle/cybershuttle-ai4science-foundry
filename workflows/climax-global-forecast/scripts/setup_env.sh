#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "ERROR: setup_env.sh failed at line %s while running: %s (exit %s).\n" "$LINENO" "$BASH_COMMAND" "$status" >&2; echo "For more command-level detail, retry with: bash -x scripts/setup_env.sh" >&2; exit "$status"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
ENV_NAME="${ENV_NAME:-climax-inference}"
ENV_PREFIX="$SCRATCH_ROOT/.conda/envs/$ENV_NAME"

if ! command -v module >/dev/null 2>&1; then
  echo "Environment modules are unavailable. Run this on a PACE ICE login/compute shell." >&2
  exit 1
fi

module purge
module load miniforge
if ! command -v conda >/dev/null 2>&1; then
  echo "The miniforge module loaded but conda is still unavailable." >&2
  echo "Check PACE's current module instructions (for example: module avail miniforge)." >&2
  exit 1
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

export CONDA_PKGS_DIRS="$SCRATCH_ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/.cache/pip"
export TMPDIR="$SCRATCH_ROOT/tmp"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export HF_HOME="$SCRATCH_ROOT/.cache/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/.cache/torch"
export GIT_CONFIG_GLOBAL="$SCRATCH_ROOT/.gitconfig"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
mkdir -p "$SCRATCH_ROOT/.conda/envs"
mkdir -p "$WORKFLOW_DIR/execution"

if command -v mamba >/dev/null 2>&1; then
  SOLVER="$(command -v mamba)"
else
  echo "mamba is unavailable; using conda (this may take longer)." >&2
  SOLVER="$(command -v conda)"
fi

if [ -f "$ENV_PREFIX/conda-meta/history" ]; then
  echo "Updating existing environment at $ENV_PREFIX"
  "$SOLVER" env update --prefix "$ENV_PREFIX" --file "$WORKFLOW_DIR/environment.yml"
else
  echo "Creating ClimaX environment at $ENV_PREFIX"
  "$SOLVER" env create --prefix "$ENV_PREFIX" --file "$WORKFLOW_DIR/environment.yml"
fi

if command -v mamba >/dev/null 2>&1; then
  mamba install --prefix "$ENV_PREFIX" --channel conda-forge "mkl=2024.0"
else
  conda install --prefix "$ENV_PREFIX" --channel conda-forge "mkl=2024.0"
fi
"$ENV_PREFIX/bin/python" -m pip install "pip==24.0"

echo "Environment prefix: $ENV_PREFIX"
"$ENV_PREFIX/bin/python" "$SCRIPT_DIR/check_environment.py" \
  --expected-prefix "$ENV_PREFIX" \
  --require-cuda
conda list --prefix "$ENV_PREFIX" --explicit > "$WORKFLOW_DIR/execution/conda-explicit.txt"
"$ENV_PREFIX/bin/python" -m pip freeze > "$WORKFLOW_DIR/execution/pip-freeze.txt"
echo "Environment creation and GPU verification completed."
echo "Activate it in this shell with: conda activate \"$ENV_PREFIX\""

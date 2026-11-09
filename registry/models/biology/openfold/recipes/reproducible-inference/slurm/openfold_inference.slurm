#!/bin/bash
#SBATCH --job-name=openfold-6kwc
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=openfold-6kwc-%j.out
#SBATCH --error=openfold-6kwc-%j.err

# Submit from the recipe directory:
#   cd registry/models/biology/openfold/recipes/reproducible-inference
#   sbatch slurm/openfold_inference.slurm

set -eo pipefail

# sbatch runs a spooled copy of this script, so locate the recipe from the
# submission directory rather than from the script path.
RECIPE_DIR="${RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -x "$RECIPE_DIR/scripts/run_inference.sh" ]]; then
  echo "ERROR: $RECIPE_DIR is not the recipe directory." >&2
  echo "Submit from recipes/reproducible-inference or set RECIPE_DIR." >&2
  exit 1
fi

# Activate the OpenFold environment. The README creates it under scratch,
# so point Conda at the same environment directory.
# Override OPENFOLD_ENV if your environment has a different name or path.
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$HOME/scratch/.conda/envs}"
if ! command -v conda >/dev/null 2>&1; then
  module load miniforge
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
OPENFOLD_ENV="${OPENFOLD_ENV:-openfold-env}"
conda activate "$OPENFOLD_ENV"

set -u

echo "=== SLURM execution metadata ==="
echo "Date: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Conda env: $CONDA_PREFIX"
echo

nvidia-smi
echo

"$RECIPE_DIR/scripts/run_inference.sh"

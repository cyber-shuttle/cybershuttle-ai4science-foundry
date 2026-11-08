#!/usr/bin/env bash
# Shared settings for the OpenFold 6KWC_1 recipe. Sourced by the other scripts.
#
# Every path has a default, so no .env file is required. To change a path,
# copy .env.example to .env (in the recipe directory) and edit it, or export
# the variable before running a script.

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$RECIPE_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$RECIPE_DIR/.env"
fi

# Large files (OpenFold source, CUDA build, weights, conda env) go in scratch,
# not in the 30 GB home directory. On PACE ICE, ~/scratch is your scratch space.
OPENFOLD_WORK="${OPENFOLD_WORK:-$HOME/scratch/openfold-6kwc}"
OPENFOLD_ENV="${OPENFOLD_ENV:-$OPENFOLD_WORK/conda-env}"
OPENFOLD_DIR="${OPENFOLD_DIR:-$OPENFOLD_WORK/openfold}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-$OPENFOLD_WORK/alignments_no_templates}"
TEMPLATE_MMCIF_DIR="${TEMPLATE_MMCIF_DIR:-$OPENFOLD_WORK/template_mmcif}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OPENFOLD_WORK/params/finetuning_no_templ_ptm_1.pt}"
FASTA_DIR="${FASTA_DIR:-$RECIPE_DIR/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RECIPE_DIR/results/run-${SLURM_JOB_ID:-interactive}}"
CONFIG_PRESET="${CONFIG_PRESET:-model_3_ptm}"

# Pinned versions used by the validated run (see docs/PROVENANCE.md).
OPENFOLD_COMMIT="e938c184a291bf053af3b14c1e3e8bb29aee57e2"   # tag v2.2.0
CUTLASS_COMMIT="bf9da7b76c766d7ee7d536afc77880a4ef1f1156"    # tag v3.6.0
CHECKPOINT_URL="https://openfold.s3.amazonaws.com/openfold_params/finetuning_no_templ_ptm_1.pt"
CHECKPOINT_SHA256="1a179937a6f61143c41f8b6a0b763ac1f4762c9dc63d2fb151240930c110bee1"

export RECIPE_DIR OPENFOLD_WORK OPENFOLD_ENV OPENFOLD_DIR ALIGNMENT_DIR \
  TEMPLATE_MMCIF_DIR CHECKPOINT_PATH FASTA_DIR OUTPUT_DIR CONFIG_PRESET

die() {
  echo "error: $*" >&2
  exit 1
}

# Load the PACE ICE modules used by the validated run.
load_modules() {
  # Lmod and conda read unset variables, so nounset is off while they run.
  set +u
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /usr/local/pace-apps/lmod/lmod/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/local/pace-apps/lmod/lmod/init/bash
    else
      die "the 'module' command is not available; run this on a PACE ICE node"
    fi
  fi
  module purge
  module load gcc/12.3.0
  module load cuda/12.1.1
  module load anaconda3
  set -u
}

# Activate the conda env. The env's gcc activation script reads unset
# variables, so nounset is turned off while it runs.
activate_env() {
  [[ -d "$OPENFOLD_ENV" ]] || die "conda env not found at $OPENFOLD_ENV; run: bash scripts/install_environment.sh"
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$OPENFOLD_ENV"
  set -u
  [[ -n "${CONDA_PREFIX:-}" ]] || die "conda activation failed for $OPENFOLD_ENV"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export TMPDIR="$OPENFOLD_WORK/tmp"
  export TRITON_CACHE_DIR="$OPENFOLD_WORK/triton-cache"
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
}

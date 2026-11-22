#!/usr/bin/env bash
#SBATCH --job-name=poyo-achilles
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${POYO_ENV:?Set POYO_ENV before submitting.}"
source "$POYO_ENV/bin/activate"
: "${POYO_REPO:?Set POYO_REPO to a clean official POYO checkout.}"
: "${ACHILLES_DATA_ROOT:?Set ACHILLES_DATA_ROOT to the existing CEBRA cache root.}"
: "${ACHILLES_TIMING:?Set ACHILLES_TIMING to verified timing evidence JSON.}"
: "${POYO_WORKFLOW:?Set POYO_WORKFLOW to this integration directory.}"
: "${POYO_OUTPUT:?Set POYO_OUTPUT to a fresh output directory.}"
: "${POYO_ACCEPT_BIN_CENTERS:?Set POYO_ACCEPT_BIN_CENTERS=yes after reviewing the approximation.}"
[[ "$POYO_ACCEPT_BIN_CENTERS" == yes ]] || exit 2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1

python "$POYO_WORKFLOW/inspect_achilles.py" --data-root "$ACHILLES_DATA_ROOT"
python "$POYO_WORKFLOW/prepare_poyo_input.py" \
    --data-root "$ACHILLES_DATA_ROOT" --timing "$ACHILLES_TIMING" \
    --output "$POYO_OUTPUT/prepared" --accept-bin-center-approximation
python "$POYO_WORKFLOW/run_poyo.py" --poyo-repo "$POYO_REPO" \
    --prepared "$POYO_OUTPUT/prepared" --output "$POYO_OUTPUT/decoding" \
    --epochs "${POYO_EPOCHS:-10}"

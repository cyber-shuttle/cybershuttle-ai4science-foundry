# Shared path defaults for SatMAE Temporal jobs.
# Source from the Foundry repo root after Slurm has set SLURM_SUBMIT_DIR.
#
# Home on ICE is 30 GB. This user was already at 96.9% of home, so every
# bulky artifact is forced onto scratch.

export SATMAE_TEMPORAL_WORKDIR="${SATMAE_TEMPORAL_WORKDIR:-$HOME/scratch/satmae-temporal}"
export SATMAE_TEMPORAL_ENV="${SATMAE_TEMPORAL_ENV:-$HOME/scratch/envs/satmae-temporal}"
export SATMAE_CODE_DIR="${SATMAE_CODE_DIR:-$SATMAE_TEMPORAL_WORKDIR/SatMAE}"
export SATMAE_UPSTREAM_COMMIT="${SATMAE_UPSTREAM_COMMIT:-0b210aceb37a14bbbd897110db5b104b3271d818}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/scratch/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/scratch/mplconfig}"

mkdir -p \
  "$SATMAE_TEMPORAL_WORKDIR"/{data,weights,results,env} \
  "$HOME/scratch/envs" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$MPLCONFIGDIR"

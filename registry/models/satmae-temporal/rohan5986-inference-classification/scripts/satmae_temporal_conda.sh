# Portable conda load for ICE. Sourced after satmae_temporal_paths.sh.
#
# Do not `module load anaconda3` if conda is already on PATH. A peer on
# miniforge hit a clash plus a conda-solved torch that failed to import
# (MKL iJIT_NotifyEvent). Prefer an existing conda (miniforge or anaconda),
# and only module-load if `conda` is missing.

satmae_load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  module load miniforge/24.3.0-0 2>/dev/null \
    || module load miniforge 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || module load anaconda 2>/dev/null \
    || true
  command -v conda >/dev/null 2>&1
}

satmae_source_conda() {
  if ! satmae_load_conda; then
    echo "conda not found. On ICE: module load miniforge   (or anaconda3)" >&2
    return 1
  fi
  # conda activate scripts are unset-variable unfriendly (MKL hook).
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set -u
}

satmae_activate_env() {
  satmae_source_conda
  if [[ ! -x "${SATMAE_TEMPORAL_ENV}/bin/python" ]]; then
    echo "missing env $SATMAE_TEMPORAL_ENV — run scripts/satmae_temporal_build_env.sbatch first" >&2
    return 1
  fi
  set +u
  conda activate "$SATMAE_TEMPORAL_ENV"
  set -u
}

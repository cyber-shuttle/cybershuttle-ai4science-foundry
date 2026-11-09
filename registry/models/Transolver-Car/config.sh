# Shared settings. Override any of these from the environment, e.g.
#   ENV_DIR=/some/path sbatch jobs/test.sbatch
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSOLVER_ROOT=${TRANSOLVER_ROOT:-$REPO_ROOT}
DATA_DIR=${DATA_DIR:-$TRANSOLVER_ROOT/data/mlcfd_data/training_data}
SAVE_DIR=${SAVE_DIR:-$TRANSOLVER_ROOT/data/preprocessed_data}
ENV_DIR=${ENV_DIR:-$HOME/scratch/envs/transolver}

# The eval code reads these
export MLCFD_DATA_DIR=$DATA_DIR
export MLCFD_SAVE_DIR=$SAVE_DIR

require_data() {
  if [ ! -d "$DATA_DIR/param8" ]; then
    echo "ERROR: dataset not found at $DATA_DIR. Please look at the README to download the dataset." >&2
    exit 1
  fi
}

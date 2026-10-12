#!/bin/bash
#SBATCH --job-name=satmae-temporal-env
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/satmae_temporal_build_env_%j.out
#SBATCH --error=logs/satmae_temporal_build_env_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_paths.sh
source "$REPO_ROOT/scripts/satmae_temporal_paths.sh"

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"
echo "workdir=$SATMAE_TEMPORAL_WORKDIR"
echo "env=$SATMAE_TEMPORAL_ENV"

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true

if [[ ! -d "$SATMAE_CODE_DIR/.git" ]]; then
  git clone https://github.com/sustainlab-group/SatMAE.git "$SATMAE_CODE_DIR"
fi
git -C "$SATMAE_CODE_DIR" fetch --all
git -C "$SATMAE_CODE_DIR" checkout "$SATMAE_UPSTREAM_COMMIT"
echo "SatMAE commit $(git -C "$SATMAE_CODE_DIR" rev-parse HEAD)"

if [[ ! -x "$SATMAE_TEMPORAL_ENV/bin/python" ]]; then
  conda env create -p "$SATMAE_TEMPORAL_ENV" -f "$REPO_ROOT/scripts/satmae_temporal_environment.yml"
else
  echo "conda prefix already exists: $SATMAE_TEMPORAL_ENV"
fi

# conda activate is unset-variable unfriendly
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SATMAE_TEMPORAL_ENV"
set -u

bash "$REPO_ROOT/scripts/satmae_temporal_patch_env.sh"

python - <<'PY'
import timm
import torch
import torchvision
print("python_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
PY

conda env export -p "$SATMAE_TEMPORAL_ENV" > "$SATMAE_TEMPORAL_WORKDIR/env/environment.lock.yml"
python -m pip freeze > "$SATMAE_TEMPORAL_WORKDIR/env/pip-freeze.txt"
echo "env build finished"

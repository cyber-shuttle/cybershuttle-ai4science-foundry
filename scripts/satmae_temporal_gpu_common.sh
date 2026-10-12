# Shared GPU-job preamble. Sourced by smoke / subset / full / random sbatch files.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_paths.sh
source "$REPO_ROOT/scripts/satmae_temporal_paths.sh"

JOB_TAG="${1:?job tag}"
echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) tag=$JOB_TAG"

nvidia-smi > "logs/${JOB_TAG}_${SLURM_JOB_ID}_nvidia-smi.txt"

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SATMAE_TEMPORAL_ENV"
set -u

python - <<'PY' | tee "logs/${JOB_TAG}_${SLURM_JOB_ID}_versions.txt"
import torch, timm, torchvision, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

export SATMAE_CODE_DIR

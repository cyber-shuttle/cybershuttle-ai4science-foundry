#!/bin/bash
#SBATCH --job-name=fcn-inference
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ---------------------------------------------------------------------------
# Single-GPU FourCastNet backbone inference + validation.
# SUBMIT FROM THE REPO ROOT:   cd ~/scratch/foundry/registry/models/climate/fourcastnet/ice-v100-rollout && sbatch --account=<acct> slurm/run_inference.sbatch
# (find your account with: sacctmgr show associations user=$USER format=account,qos,partition)
# (Slurm copies this script to a spool dir, so paths are built from
#  $SLURM_SUBMIT_DIR, never from $0; --output paths are relative to the submit dir.)
#
# What runs: one rollout starting at index 0 of the 2018 out-of-sample file
# (--vis makes inference.py use a single initial condition), 40 saved states:
# the initial condition (lead 0 h) plus 39 six-hour steps, ending at lead 234 h.
# Note: no `set -u`, because conda's activate scripts reference unset variables.
# ---------------------------------------------------------------------------

set -eo pipefail

REPO="$SLURM_SUBMIT_DIR"
FOURCASTNET_REPO="${FOURCASTNET_REPO:-$HOME/FourCastNet}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/scratch/fcn-artifacts}"
OUTPUT_DIR="$HOME/scratch/fcn-run-$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR" "$REPO/logs" "$REPO/outputs"

# Stop early if the FourCastNet checkout is not the commit this workflow was validated on.
PINNED_COMMIT=93360c1720a9f97aabf970689f21c9fad8737788
ACTUAL_COMMIT="$(git -C "$FOURCASTNET_REPO" rev-parse HEAD 2>/dev/null || echo missing)"
if [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then
  echo "ERROR: $FOURCASTNET_REPO is at $ACTUAL_COMMIT but this workflow expects $PINNED_COMMIT." >&2
  echo "Fix: git -C $FOURCASTNET_REPO checkout $PINNED_COMMIT" >&2
  exit 1
fi

module purge
module load miniforge
module load miniforge
module load miniforge      # repeated on PACE-ICE to override the default anaconda module
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fourcastnet-inference

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FOURCASTNET_REPO:${PYTHONPATH:-}"

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "repo commit (this repo):        $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "repo commit (FourCastNet):      $(git -C "$FOURCASTNET_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
python -c "import torch,numpy; print('torch',torch.__version__,'numpy',numpy.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python "$REPO/scripts/configure_paths.py" \
  --fourcastnet-repo "$FOURCASTNET_REPO" \
  --artifact-root "$ARTIFACT_ROOT"
git -C "$FOURCASTNET_REPO" diff config/AFNO.yaml > "$REPO/logs/config_patch.diff" || true

# Sample GPU memory/utilization twice a second for the whole run.
GPU_CSV="$REPO/logs/gpu_usage_$SLURM_JOB_ID.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500 > "$GPU_CSV" &
SMI_PID=$!

echo "=== Running inference ==="
START=$(date +%s)
cd "$FOURCASTNET_REPO"
python inference/inference.py \
  --config=afno_backbone \
  --run_num=0 \
  --weights "$ARTIFACT_ROOT/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt" \
  --override_dir "$OUTPUT_DIR" \
  --vis
END=$(date +%s)
kill "$SMI_PID" 2>/dev/null || true
cd "$REPO"

echo "=== Inference wall time: $((END-START)) s ==="
awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print "=== Peak GPU memory: " m " MiB ==="}' "$GPU_CSV"
echo "=== Output files ==="
ls -la "$OUTPUT_DIR"

echo "=== Validation ==="
set +e
python "$REPO/scripts/validate.py" --run-dir "$OUTPUT_DIR" --plot-out "$REPO/outputs/skill_vs_leadtime.png" \
  --csv-out "$REPO/outputs/z500_skill_$SLURM_JOB_ID.csv" \
  --reference-csv "$REPO/outputs/z500_skill_6205007.csv"
RC=$?
set -e

cp "$OUTPUT_DIR/inference_out.log" "$REPO/logs/inference_out_$SLURM_JOB_ID.log" 2>/dev/null || true
# The prediction .h5 is ~6.6 GB (it stores predicted and ground-truth fields); it stays on scratch.
echo "=== Done (validation exit code $RC) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit $RC

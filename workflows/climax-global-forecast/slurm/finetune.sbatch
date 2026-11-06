#!/bin/bash
#SBATCH --job-name=climax-finetune
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/climax-finetune-%j.out
#SBATCH --error=logs/climax-finetune-%j.err

set -euo pipefail

WORKFLOW_DIR="${WORKFLOW_DIR:-$HOME/scratch/climax-inference/workflows/climax-global-forecast}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/climax-reproduction}"
ENV_NAME="${ENV_NAME:-climax-inference}"
CLIMAX_SOURCE="${CLIMAX_SOURCE:-$PROJECT_DIR/ClimaX}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$PROJECT_DIR/checkpoints/5.625deg.ckpt}"
ERA5_NPZ="${ERA5_NPZ:-$PROJECT_DIR/data/5.625deg_npz}"
TRAIN_DIR="${TRAIN_DIR:-$PROJECT_DIR/runs/global-forecast}"
FORECAST_VARIABLE="${FORECAST_VARIABLE:-}"
QUICK_TUNE_STEPS="${QUICK_TUNE_STEPS:-0}"

cd "$WORKFLOW_DIR"
mkdir -p logs "$TRAIN_DIR"

set +u
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$HOME/scratch/.conda/pkgs"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export TORCH_HOME="$HOME/scratch/.cache/torch"
export GIT_CONFIG_GLOBAL="$HOME/scratch/.gitconfig"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
conda activate "$ENV_NAME"
set -u

for required_path in "$CLIMAX_SOURCE/src/climax/global_forecast/train.py" "$PRETRAINED_CHECKPOINT" "$ERA5_NPZ"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done
export PYTHONPATH="$CLIMAX_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import climax; print("ClimaX package:", climax.__file__)'
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$PRETRAINED_CHECKPOINT" \
  | sha256sum --check -

export OUTPUT_DIR="$TRAIN_DIR"
export CLIMAX_COMMIT
CLIMAX_COMMIT="$(git -C "$CLIMAX_SOURCE" rev-parse HEAD)"
export WORKFLOW_COMMIT
WORKFLOW_COMMIT="$(git -C "$WORKFLOW_DIR/../.." rev-parse HEAD)"
TRAIN_ARGS=(
  --config "$CLIMAX_SOURCE/configs/global_forecast_climax.yaml"
  --trainer.default_root_dir="$TRAIN_DIR"
  --trainer.strategy=single_device
  --trainer.devices=1
  --trainer.max_epochs=50
  --trainer.precision=16
  --data.root_dir="$ERA5_NPZ"
  --data.predict_range=72
  --data.batch_size=4
  --data.num_workers=0
  --model.pretrained_path="$PRETRAINED_CHECKPOINT"
  --model.lr=5e-7
  --model.beta_1=0.9
  --model.beta_2=0.99
  --model.weight_decay=1e-5
)
if [ -n "$FORECAST_VARIABLE" ]; then
  if [ "$FORECAST_VARIABLE" != "geopotential_500" ]; then
    echo "This reduced-data workflow supports FORECAST_VARIABLE=geopotential_500." >&2
    exit 1
  fi
  TRAIN_ARGS+=(
    "--data.variables=[\"$FORECAST_VARIABLE\"]"
    "--data.out_variables=[\"$FORECAST_VARIABLE\"]"
  )
fi
if [ "$QUICK_TUNE_STEPS" != "0" ]; then
  if ! [[ "$QUICK_TUNE_STEPS" =~ ^[0-9]+$ ]] || [ "$QUICK_TUNE_STEPS" -lt 25 ]; then
    echo "QUICK_TUNE_STEPS must be 0 (full training) or an integer of at least 25." >&2
    exit 1
  fi
  TRAIN_ARGS+=(
    "--trainer.max_steps=$QUICK_TUNE_STEPS"
    "--trainer.limit_train_batches=$QUICK_TUNE_STEPS"
    --model.warmup_epochs=1
    "--model.max_epochs=$QUICK_TUNE_STEPS"
    --trainer.val_check_interval=1.0
    --trainer.limit_val_batches=8
    --trainer.limit_test_batches=8
    --trainer.num_sanity_val_steps=0
  )
  echo "quick_tune_steps=$QUICK_TUNE_STEPS"
  echo "quick_tune_validation_batches=8"
  echo "quick_tune_test_batches=8"
fi
if [ "$#" -gt 0 ]; then
  RESUME_CHECKPOINT="$1"
  if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "Resume checkpoint is missing: $RESUME_CHECKPOINT" >&2
    exit 1
  fi
  TRAIN_ARGS+=("--trainer.resume_from_checkpoint=$RESUME_CHECKPOINT")
fi

echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "node=$(hostname)"
echo "climax_commit=$CLIMAX_COMMIT"
echo "workflow_commit=$WORKFLOW_COMMIT"
echo "pretrained_checkpoint=$PRETRAINED_CHECKPOINT"
echo "era5_data=$ERA5_NPZ"
echo "train_output=$TRAIN_DIR"
nvidia-smi

python "$CLIMAX_SOURCE/src/climax/global_forecast/train.py" "${TRAIN_ARGS[@]}"

nvidia-smi

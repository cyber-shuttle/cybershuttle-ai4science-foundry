#!/bin/bash
#SBATCH --job-name=climax-inference
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/climax-inference-%j.out
#SBATCH --error=logs/climax-inference-%j.err

set -euo pipefail

WORKFLOW_DIR="${WORKFLOW_DIR:-$HOME/scratch/climax-inference/workflows/climax-global-forecast}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/climax-reproduction}"
ENV_NAME="${ENV_NAME:-climax-inference}"
CLIMAX_SOURCE="${CLIMAX_SOURCE:-$PROJECT_DIR/ClimaX}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$PROJECT_DIR/checkpoints/5.625deg.ckpt}"
FINETUNED_CHECKPOINT="${1:?Usage: sbatch slurm/infer.sbatch /path/to/validation-selected.ckpt}"
ERA5_NPZ="${ERA5_NPZ:-$PROJECT_DIR/data/5.625deg_npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/runs/inference-${SLURM_JOB_ID:-manual}}"

cd "$WORKFLOW_DIR"
mkdir -p logs "$OUTPUT_DIR"
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
CLIMAX_PYTHON="$(command -v python)"
export CLIMAX_COMMIT
CLIMAX_COMMIT="$(git -C "$CLIMAX_SOURCE" rev-parse HEAD)"
export WORKFLOW_COMMIT
WORKFLOW_COMMIT="$(git -C "$WORKFLOW_DIR/../.." rev-parse HEAD)"

echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "node=$(hostname)"
echo "climax_commit=$CLIMAX_COMMIT"
echo "workflow_commit=$WORKFLOW_COMMIT"
echo "python=$CLIMAX_PYTHON"
echo "fine_tuned_checkpoint=$FINETUNED_CHECKPOINT"
echo "pretrained_checkpoint=$PRETRAINED_CHECKPOINT"
echo "data=$ERA5_NPZ"
nvidia-smi

"$CLIMAX_PYTHON" scripts/infer.py \
  --climax-source "$CLIMAX_SOURCE" \
  --config "$CLIMAX_SOURCE/configs/global_forecast_climax.yaml" \
  --checkpoint "$FINETUNED_CHECKPOINT" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --data-dir "$ERA5_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --time-index 0 \
  --lead-hours 72 \
  --num-samples 32 \
  --sample-stride 24 \
  --device cuda

nvidia-smi

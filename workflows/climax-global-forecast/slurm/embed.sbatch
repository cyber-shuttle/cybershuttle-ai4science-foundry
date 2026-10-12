#!/bin/bash
#SBATCH --job-name=climax-embed
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/climax-embed-%j.out
#SBATCH --error=logs/climax-embed-%j.err

set -euo pipefail

WORKFLOW_DIR="${WORKFLOW_DIR:-$HOME/scratch/climax-inference/workflows/climax-global-forecast}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/climax-reproduction}"
ENV_NAME="${ENV_NAME:-climax-inference}"
CLIMAX_SOURCE="${CLIMAX_SOURCE:-$PROJECT_DIR/ClimaX}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/checkpoints/5.625deg.ckpt}"
ERA5_NPZ="${ERA5_NPZ:-$PROJECT_DIR/data/geopotential_npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/runs/embeddings-${SLURM_JOB_ID:-manual}}"

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

echo "job_id=${SLURM_JOB_ID:-not-set}"
echo "node=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "climax_commit=$(git -C "$CLIMAX_SOURCE" rev-parse HEAD)"
echo "workflow_commit=$(git -C "$WORKFLOW_DIR/../.." rev-parse HEAD)"
echo "checkpoint=$CHECKPOINT"
echo "data=$ERA5_NPZ"
nvidia-smi

"$CLIMAX_PYTHON" scripts/embed.py \
  --climax-source "$CLIMAX_SOURCE" \
  --config "$CLIMAX_SOURCE/configs/pretrain_climax.yaml" \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$ERA5_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --variable geopotential_500 \
  --split test \
  --time-index 0 \
  --num-samples 32 \
  --sample-stride 24 \
  --device cuda

echo "end=$(date --iso-8601=seconds)"
nvidia-smi

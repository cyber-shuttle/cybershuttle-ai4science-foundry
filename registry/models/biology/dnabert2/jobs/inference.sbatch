#!/bin/bash
#SBATCH --job-name=dnabert2-inference
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Move to project directory
cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SLURM_SUBMIT_DIR/.conda-env"

# Put Hugging Face downloads on scratch
export HF_HOME="$SLURM_SUBMIT_DIR/.hf-cache"
export TMPDIR="$SLURM_SUBMIT_DIR/.tmp"
mkdir -p "$HF_HOME" "$TMPDIR"

# Record execution details
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date -Is)"

nvidia-smi

# Execute DNABERT-2 inference
python -u scripts/inference.py

echo "End time: $(date -Is)"

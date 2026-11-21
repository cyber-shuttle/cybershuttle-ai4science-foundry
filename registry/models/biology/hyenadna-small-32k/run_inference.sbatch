#!/bin/bash
#SBATCH -J hyenadna-infer
# Set your account and partition before submitting (see README):
#   export SBATCH_ACCOUNT=<account>  SBATCH_PARTITION=<partition>
#SBATCH --gres=gpu:1
#SBATCH -C nvidia-gpu
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH_DIR/.conda/envs/hyenadna_env}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.hf_cache}"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "ERROR: no Python environment at $ENV_PREFIX"
    echo "Run 'bash setup_env.sh' first. If you built the env somewhere else, run"
    echo "'export ENV_PREFIX=/your/env/path' before sbatch."
    exit 1
fi

module load miniforge
set +u   # conda's activate/deactivate scripts reference unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

export OUT_DIR="outputs/run_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Env:     $ENV_PREFIX"
echo "Output:  $OUT_DIR"
nvidia-smi
pip freeze > "logs/pip_freeze_${SLURM_JOB_ID}.txt"

python infer.py
python validate.py

echo "Finished: $(date)"
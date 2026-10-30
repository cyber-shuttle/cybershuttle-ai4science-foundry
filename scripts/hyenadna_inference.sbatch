#!/bin/bash
#SBATCH --job-name=hyenadna-inference
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hyenadna-inference-%j.out
#
# Reproducible HyenaDNA inference job for PACE ICE (ice-gpu partition).
# Builds the environment (idempotent — skips install if already present),
# runs the smoke test + validation check, then embeds the requested input.
#
# Usage: sbatch hyenadna_inference.sbatch [path/to/input.fasta]

set -euo pipefail
mkdir -p logs

REPO_DIR="$HOME/scratch/hyena-dna"
ENV_PREFIX="$HOME/scratch/.conda/envs/hyena-dna"
# Not ${BASH_SOURCE[0]}'s dirname: sbatch copies this script into a per-job
# spool directory on the compute node before running it, so that would
# resolve to the spool dir, not to where the sibling scripts actually live
# (confirmed live in the training sbatch, which has this same pattern).
# SLURM_SUBMIT_DIR is the directory `sbatch` was invoked from, which is
# where this file's siblings are.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# module purge first: a fresh job starts with anaconda3 loaded by default,
# and `module load miniforge` on top of that triggers Lmod's automatic
# swap-out of anaconda3, whose deactivation hook references an unset
# variable -- fatal under `set -u`. Found live while testing the training
# sbatch job (which has this same pattern); fixed here too since this
# script was never actually run as a real `sbatch` submission before now.
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    # Must be sourced, not run via `bash`: conda activation and the
    # cache-redirect exports (GIT_CONFIG_GLOBAL, PIP_CACHE_DIR, TMPDIR,
    # HF_HOME) only persist in the shell that performs them. Sourcing it here
    # also activates the env, so the explicit `conda activate` below only
    # runs on the already-set-up path.
    source "$SCRIPT_DIR/hyenadna_setup_env.sh" "$ENV_PREFIX" "$REPO_DIR"
else
    conda activate "$ENV_PREFIX"
fi

export HYENADNA_REPO="$REPO_DIR"
export HF_HOME="$HOME/scratch/.cache/huggingface"  # must match setup script, or the checkpoint re-downloads

python "$SCRIPT_DIR/hyenadna_validate.py" --model hyenadna-tiny-1k-seqlen

if [ -n "${1:-}" ]; then
    python "$SCRIPT_DIR/hyenadna_inference.py" \
        --model hyenadna-tiny-1k-seqlen \
        --fasta "$1" \
        --output "results/hyenadna-embeddings-${SLURM_JOB_ID}.pt"
fi

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
# Not SLURM_SUBMIT_DIR either: that's wherever `sbatch` was invoked FROM,
# which only equals this file's directory if the caller happens to cd into
# it first -- confirmed live: `sbatch scripts/hyenadna_inference.sbatch` run
# from a repo checkout's root resolved SCRIPT_DIR to the repo root and died
# on "can't open file '.../hyenadna_validate.py'", since that file is in
# scripts/, not the root. `scontrol show job` still has this job's real
# Command= (the path as submitted, not the spool copy) while it's running,
# so derive SCRIPT_DIR from that; SLURM_SUBMIT_DIR is only a fallback.
SCRIPT_DIR="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'Command=\K\S+' | xargs -r dirname)"
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
fi

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
    # torch.save has no "create the parent dir for me" option -- confirmed
    # live: without this, a fresh job with no prior results/ dir dies on
    # the save call after a full successful embedding pass.
    mkdir -p results
    python "$SCRIPT_DIR/hyenadna_inference.py" \
        --model hyenadna-tiny-1k-seqlen \
        --fasta "$1" \
        --output "results/hyenadna-embeddings-${SLURM_JOB_ID}.pt"
fi

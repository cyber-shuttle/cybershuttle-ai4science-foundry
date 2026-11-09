#!/bin/bash
#SBATCH --job-name=satmae-temporal-weights
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/satmae_temporal_download_weights_%j.out
#SBATCH --error=logs/satmae_temporal_download_weights_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_paths.sh
source "$REPO_ROOT/scripts/satmae_temporal_paths.sh"
bash "$REPO_ROOT/scripts/satmae_temporal_download_weights.sh"

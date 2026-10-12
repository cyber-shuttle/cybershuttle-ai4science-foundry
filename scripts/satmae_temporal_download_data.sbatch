#!/bin/bash
#SBATCH --job-name=satmae-temporal-data
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/satmae_temporal_download_data_%j.out
#SBATCH --error=logs/satmae_temporal_download_data_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs
# shellcheck source=satmae_temporal_paths.sh
source "$REPO_ROOT/scripts/satmae_temporal_paths.sh"

module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || true
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -x "$SATMAE_TEMPORAL_ENV/bin/python" ]]; then
  conda activate "$SATMAE_TEMPORAL_ENV"
fi
set -u

bash "$REPO_ROOT/scripts/satmae_temporal_download_data.sh"
python "$REPO_ROOT/scripts/satmae_temporal_prepare_csv.py" \
  --csv "$SATMAE_TEMPORAL_WORKDIR/data/val_62classes.csv" \
  --data-root "$SATMAE_TEMPORAL_WORKDIR/data" \
  --out "$SATMAE_TEMPORAL_WORKDIR/data/val_prepared.csv" \
  --subset-out "$SATMAE_TEMPORAL_WORKDIR/data/val_subset500.csv" \
  --subset-size 500

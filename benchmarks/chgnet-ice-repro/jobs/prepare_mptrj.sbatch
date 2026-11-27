#!/bin/bash
#SBATCH --job-name=chgnet-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mptrj-prep-%j.out
#SBATCH --error=logs/mptrj-prep-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate chgnet-repro

mkdir -p "$HOME/scratch/chgnet-mptrj" inputs

echo "Starting MPtrj preparation"
date -u

python scripts/download_mptrj.py \
  --output "$HOME/scratch/chgnet-mptrj/mptrj_source"
cp "$HOME/scratch/chgnet-mptrj/mptrj_source.manifest.json" \
  docs/mptrj_source_manifest.json

python scripts/make_subset.py \
  --input "$HOME/scratch/chgnet-mptrj/mptrj_source" \
  --output inputs/mptrj_subset.json \
  --count 24 \
  --seed 42 \
  --min-atoms 2 \
  --max-atoms 64

sha256sum inputs/mptrj_subset.json \
  > inputs/mptrj_subset.sha256

echo "Data preparation completed"
date -u

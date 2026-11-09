#!/bin/bash
#SBATCH --job-name=clay-download
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/logs/%x-%j.out
#SBATCH --error=benchmarks/logs/%x-%j.err
#
# Step 2 of the Clay workflow: download and checksum the checkpoint, the
# DINOv2 teacher, the NAIP tile, and Clay's reference embeddings (~6.6 GB in
# total, all under $CLAY_WORKDIR). Needs the env from step 1.
#
# Submit from the Clay model folder:
#   sbatch scripts/clay_download.sbatch
# or chain it after the env build:
#   sbatch --dependency=afterok:<build job id> scripts/clay_download.sbatch

set -euo pipefail
trap 'echo "== clay-download FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR

: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Clay model folder.}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root

echo "== clay-download  job=${SLURM_JOB_ID:-none}  node=$(hostname)  start=$(date -Is)"
echo "== CLAY_WORKDIR=$CLAY_WORKDIR"
echo "== $(clay_code_version)"

echo "== Network check"
clay_check_url "https://huggingface.co/api/models/made-with-clay/Clay"
clay_check_url "${NAIP_STAC_API}/collections/naip"
clay_check_url "$REF_URL"

# Start a fresh checksum record, so it lists exactly the files of this run
# (the copy committed in the repo is from an earlier run).
rm -f "$CLAY_EVIDENCE/outputs/checksums.sha256"
bash scripts/clay_download_weights.sh
bash scripts/clay_download_data.sh

echo "== Disk use"
du -sh "$CLAY_WEIGHTS" "$CLAY_DATA" "$HF_HOME" | sed 's/^/   /'
echo "== DOWNLOAD COMPLETED  end=$(date -Is)"

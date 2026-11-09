#!/bin/bash
#SBATCH --job-name=clay-smoke
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/%x-%j.out
#SBATCH --error=benchmarks/logs/%x-%j.err
#
# Step 3 of the Clay workflow: a quick GPU smoke test on the first 8 chips.
# Checks that the checkpoint loads strictly, the output is [8, 1024] and
# finite, two identical runs agree, and different chips differ.
# Needs steps 1-2 (env build and downloads).
#
# Submit from the Clay model folder:
#   sbatch scripts/clay_smoke_test.sbatch
#
# Writes:
#   $CLAY_WORKDIR/results/smoke-<jobid>/run{1,2}.parquet   (embeddings)
#   benchmarks/outputs/smoke/validation_report.json

set -euo pipefail
trap 'echo "== clay-smoke FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR

: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Clay model folder.}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
clay_make_dirs
EVIDENCE="$CLAY_EVIDENCE/outputs/smoke"
RUN="$CLAY_RESULTS/smoke-${SLURM_JOB_ID}"

echo "== clay-smoke  job=${SLURM_JOB_ID}  node=$(hostname)  start=$(date -Is)"
echo "== $(clay_code_version)"
rm -rf "$EVIDENCE"                  # results of an earlier run are replaced, not mixed
clay_gpu_setup

for i in 1 2; do
    echo "== Embedding run $i (8 chips)"
    python scripts/clay_embed.py "${CLAY_EMBED_INPUTS[@]}" --limit 8 --batch-size 8 --out "$RUN/run$i.parquet"
done

echo "== Smoke checks"
python scripts/clay_validate.py --smoke --embeddings "$RUN/run1.parquet" --repeat "$RUN/run2.parquet" \
    --out-dir "$EVIDENCE"

echo "== Slurm accounting"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%40,State,Elapsed,MaxRSS
echo "== SMOKE TEST COMPLETED  end=$(date -Is)"

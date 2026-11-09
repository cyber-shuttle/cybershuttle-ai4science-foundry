#!/bin/bash
#SBATCH --job-name=clay-inference
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/logs/%x-%j.out
#SBATCH --error=benchmarks/logs/%x-%j.err
#
# Step 4 of the Clay workflow: embed all 725 chips of the NAIP tile with
# Clay v1.5 and compare each chip with Clay's published embedding of the same
# chip (pass criterion: mean cosine similarity >= 0.99). Needs steps 1-3.
#
# Submit from the Clay model folder:
#   sbatch scripts/clay_inference.sbatch
#
# Writes:
#   $CLAY_WORKDIR/results/inference-<jobid>/embeddings.parquet   (725 x 1024 embeddings)
#   benchmarks/outputs/inference/
#       validation_report.json, validation_cosine.png   checks and histogram
#       embed_run.json                                  inputs, versions, GPU, timing
#       embeddings_sample.parquet                       the first 16 chips
#       summary.txt                                     one-line result

set -euo pipefail
trap 'echo "== clay-inference FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR

: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Clay model folder.}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root
clay_make_dirs
EVIDENCE="$CLAY_EVIDENCE/outputs/inference"
RUN="$CLAY_RESULTS/inference-${SLURM_JOB_ID}"

echo "== clay-inference  job=${SLURM_JOB_ID}  node=$(hostname)  start=$(date -Is)"
echo "== $(clay_code_version)"
rm -rf "$EVIDENCE" && mkdir -p "$EVIDENCE"   # results of an earlier run are replaced, not mixed
clay_gpu_setup

echo "== Embedding all 725 chips"
python scripts/clay_embed.py "${CLAY_EMBED_INPUTS[@]}" --batch-size 16 --out "$RUN/embeddings.parquet"
cp "$RUN/embeddings.json" "$EVIDENCE/embed_run.json"

echo "== Validation against Clay's published embeddings"
RESULT=PASS
python scripts/clay_validate.py --embeddings "$RUN/embeddings.parquet" --reference "$CLAY_DATA/$REF_NAME" \
    --out-dir "$EVIDENCE" --label "our embeddings vs Clay's published embeddings" || RESULT=FAIL

echo "== Embeddings sample (first 16 chips, kept as an output artifact)"
python - "$RUN/embeddings.parquet" "$EVIDENCE/embeddings_sample.parquet" <<'PY'
import sys
import pyarrow.parquet as pq
table = pq.read_table(sys.argv[1])
pq.write_table(table.slice(0, 16), sys.argv[2])
print(f"   wrote {sys.argv[2]}: 16 of {table.num_rows} rows")
PY

python - "$EVIDENCE" "$SLURM_JOB_ID" <<'PY' | tee "$EVIDENCE/summary.txt"
import json, subprocess, sys
from pathlib import Path
evidence, job = Path(sys.argv[1]), sys.argv[2]
tree = subprocess.run(["git", "rev-parse", "HEAD:./scripts"], capture_output=True, text=True).stdout.strip()
r = json.loads((evidence / "validation_report.json").read_text())
print(f"clay-inference job {job}, scripts tree {tree}")
print(f"{r['n_chips']} chips, ours vs Clay's published embeddings: "
      f"mean cosine {r['matched_cosine']['mean']:.8f} (min {r['matched_cosine']['min']:.8f}), "
      f"mismatched control {r['mismatched_cosine_control']['mean']:.4f}, "
      f"retrieval {r['retrieval_top1_accuracy']:.1%}, "
      f"relative L2 {r['relative_l2_error']['mean']:.2e}: {r['result']}")
PY

echo "== Scratch space used by the whole workflow"
du -sh "$CLAY_WORKDIR" "$CLAY_WORKDIR"/* 2>/dev/null | sed 's/^/   /'

echo "== Slurm accounting"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%40,State,Elapsed,MaxRSS
echo "== WORKFLOW COMPLETED  validation: $RESULT  end=$(date -Is)"

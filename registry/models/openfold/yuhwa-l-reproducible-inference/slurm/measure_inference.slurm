#!/bin/bash
#SBATCH --job-name=openfold-measure
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:a100:1
# The baseline in results/measurements/ was measured on an A100 40GB. Time and
# memory depend on the card, so always measure on that same type.
#SBATCH --constraint=A100-40GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=openfold-measure-%j.out

# Measures inference time, GPU memory and use, and accuracy for 6KWC and the
# five length-series proteins, one after another on one A100 (baseline
# settings). Submit from the recipe directory after README step 6.1:
#   sbatch slurm/measure_inference.slurm
# Results: results/measurements-<job ID>/ (set OUTPUT_DIR to write somewhere else).

set -eo pipefail

# sbatch runs a spooled copy of this script, so locate the recipe from the
# submission directory rather than from the script path.
RECIPE_DIR="${RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$RECIPE_DIR/scripts/measure_inference.py" ]]; then
  echo "ERROR: $RECIPE_DIR is not the recipe directory." >&2
  exit 1
fi
# A new folder per job, so the committed baseline in results/measurements/ is kept.
OUTPUT_DIR="${OUTPUT_DIR:-$RECIPE_DIR/results/measurements-${SLURM_JOB_ID:-local}}"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$HOME/scratch/.conda/envs}"
if ! command -v conda >/dev/null 2>&1; then
  module load miniforge
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENFOLD_ENV:-openfold-env}"
set -u
# OpenFold's compiled attention kernel links against libraries in the env.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Keep compiled-kernel caches out of the small $HOME quota.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/.triton-cache}"

echo "Date: $(date -Is)  Job: ${SLURM_JOB_ID:-none}  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

cd "$RECIPE_DIR"
status=0
measure() {  # FASTA ALIGNMENT_DIR REFERENCE RELEASE_DATE
  # Templates must be older than the experimental structure, so the
  # prediction cannot copy it: use the day before it was released.
  python scripts/measure_inference.py --fasta "$1" --alignment-dir "$2" \
    --reference "$3" --reference-chain A \
    --max-template-date "$(date -d "$4 -1 day" +%F)" \
    --output-dir "$OUTPUT_DIR" || status=1
}

# 6KWC (released 2021-01-27), then the length series from short to long.
measure inputs/6KWC.fasta alignments reference/6KWC.pdb 2021-01-27
while IFS=$'\t' read -r id chain length released _; do
  measure "length-series/inputs/${id}_${chain}.fasta" length-series/alignments \
    "length-series/reference/${id}.cif" "$released"
done < <(tail -n +2 length-series/targets.tsv | sort -t$'\t' -k3,3n)

echo
column -s, -t "$OUTPUT_DIR/measurements.csv"
exit $status

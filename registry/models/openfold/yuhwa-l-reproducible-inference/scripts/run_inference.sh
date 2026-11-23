#!/bin/bash
set -euo pipefail

# Reproducible OpenFold inference for the six proteins in proteins.tsv.
#
# Usage (inside a GPU allocation, OpenFold environment active):
#   ./scripts/run_inference.sh            # all six proteins, shortest first
#   ./scripts/run_inference.sh 6KWC_1     # only the listed proteins
#
# Each protein is predicted with its own --max_template_date (the day before
# its experimental structure was released), so OpenFold cannot use that
# structure as a template.
#
# Requirements:
#   - Official OpenFold checkout
#   - OpenFold-compatible Conda environment
#   - NVIDIA GPU
#   - AlphaFold model parameters and template database
#   - Precomputed alignments, unpacked (README step 2.1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENFOLD_DIR="${OPENFOLD_DIR:-$HOME/scratch/openfold}"

PARAMS="${PARAMS:-/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/params/params_model_1_ptm.npz}"

TEMPLATE_DIR="${TEMPLATE_DIR:-/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb_mmcif/mmcif_files}"

OUTPUT_DIR="${OUTPUT_DIR:-$RECIPE_DIR/results/run}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/.triton-cache}"

# Preflight checks: fail early with a clear message.
fail() { echo "ERROR: $*" >&2; exit 1; }

cd "$RECIPE_DIR"
targets=()
while IFS=$'\t' read -r target _ released fasta alignments _; do
  if [[ $# -gt 0 && ! " $* " == *" $target "* ]]; then
    continue
  fi
  [[ -f "$fasta" ]] || fail "Missing input FASTA: $RECIPE_DIR/$fasta"
  [[ -f "$alignments/$target/bfd_uniref_hits.a3m" ]] \
    || fail "Missing alignments: $RECIPE_DIR/$alignments/$target (unpack them first, README step 2.1)."
  targets+=("$target|$released|$fasta|$alignments")
done < <(tail -n +2 proteins.tsv)
[[ ${#targets[@]} -gt 0 ]] || fail "No protein in proteins.tsv matches: $*"

[[ -f "$OPENFOLD_DIR/run_pretrained_openfold.py" ]] \
  || fail "OpenFold checkout not found at $OPENFOLD_DIR (set OPENFOLD_DIR)."
[[ -f "$PARAMS" ]] \
  || fail "Model parameters not found: $PARAMS (set PARAMS)."
[[ -d "$TEMPLATE_DIR" ]] \
  || fail "Template mmCIF directory not found: $TEMPLATE_DIR (set TEMPLATE_DIR)."
python -c "import torch, openfold" 2>/dev/null \
  || fail "Cannot import torch/openfold; activate the OpenFold Conda environment first."
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || fail "No CUDA GPU visible; run inside a GPU allocation (see slurm/openfold_inference.slurm)."

# OpenFold's compiled attention kernel links against libraries in the Conda env.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

echo "=== OpenFold reproducible inference ==="
echo "OpenFold:       $OPENFOLD_DIR"
echo "Parameters:     $PARAMS"
echo "Templates:      $TEMPLATE_DIR"
echo "Output:         $OUTPUT_DIR"
echo "Proteins:       ${#targets[@]}"
echo

for entry in "${targets[@]}"; do
  IFS='|' read -r target released fasta alignments <<< "$entry"
  run_dir="$OUTPUT_DIR/$target"
  # OpenFold predicts every FASTA in its input folder, so give each protein its own.
  mkdir -p "$run_dir/input"
  cp "$RECIPE_DIR/$fasta" "$run_dir/input/$target.fasta"
  template_date="$(date -d "$released -1 day" +%F)"
  echo "=== $target (templates up to $template_date) -> $run_dir ==="
  start=$SECONDS
  (cd "$OPENFOLD_DIR" && python run_pretrained_openfold.py \
    "$run_dir/input" \
    "$TEMPLATE_DIR" \
    --use_precomputed_alignments "$RECIPE_DIR/$alignments" \
    --output_dir "$run_dir" \
    --config_preset model_1_ptm \
    --jax_param_path "$PARAMS" \
    --model_device cuda:0 \
    --data_random_seed 42 \
    --max_template_date "$template_date" \
    --skip_relaxation)
  echo "=== $target done in $((SECONDS - start)) s ==="
  echo
done

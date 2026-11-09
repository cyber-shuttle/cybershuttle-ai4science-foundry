#!/bin/bash
set -euo pipefail

# Reproducible OpenFold inference for PDB target 6KWC.
#
# Requirements:
#   - Official OpenFold checkout
#   - OpenFold-compatible Conda environment
#   - NVIDIA GPU
#   - AlphaFold model parameters and template database
#   - Precomputed alignments for 6KWC_1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENFOLD_DIR="${OPENFOLD_DIR:-$HOME/scratch/openfold}"

PARAMS="${PARAMS:-/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/params/params_model_1_ptm.npz}"

TEMPLATE_DIR="${TEMPLATE_DIR:-/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb_mmcif/mmcif_files}"

ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$RECIPE_DIR/alignments}"

OUTPUT_DIR="${OUTPUT_DIR:-$RECIPE_DIR/results/run}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/.triton-cache}"

# Preflight checks: fail early with a clear message.
fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$OPENFOLD_DIR/run_pretrained_openfold.py" ]] \
  || fail "OpenFold checkout not found at $OPENFOLD_DIR (set OPENFOLD_DIR)."
[[ -f "$RECIPE_DIR/inputs/6KWC.fasta" ]] \
  || fail "Missing input FASTA: $RECIPE_DIR/inputs/6KWC.fasta"
[[ -d "$ALIGNMENTS_DIR/6KWC_1" ]] \
  || fail "Missing precomputed alignments: $ALIGNMENTS_DIR/6KWC_1"
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

mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== OpenFold reproducible inference ==="
echo "OpenFold:       $OPENFOLD_DIR"
echo "Input:          $RECIPE_DIR/inputs"
echo "Alignments:     $ALIGNMENTS_DIR"
echo "Parameters:     $PARAMS"
echo "Templates:      $TEMPLATE_DIR"
echo "Output:         $OUTPUT_DIR"
echo

cd "$OPENFOLD_DIR"

python run_pretrained_openfold.py \
  "$RECIPE_DIR/inputs" \
  "$TEMPLATE_DIR" \
  --use_precomputed_alignments "$ALIGNMENTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --config_preset model_1_ptm \
  --jax_param_path "$PARAMS" \
  --model_device cuda:0 \
  --data_random_seed 42 \
  --skip_relaxation

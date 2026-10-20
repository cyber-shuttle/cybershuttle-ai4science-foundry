#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "missing .env"
  exit 1
fi

# shellcheck disable=SC1091
source .env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"

cd "$OPENFOLD_DIR"

python run_pretrained_openfold.py "$FASTA_DIR" "$TEMPLATE_MMCIF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_precomputed_alignments "$ALIGNMENT_DIR" \
  --config_preset "$CONFIG_PRESET" \
  --openfold_checkpoint_path "$CHECKPOINT_PATH" \
  --model_device "cuda:0" \
  --data_random_seed 0 \
  --skip_relaxation

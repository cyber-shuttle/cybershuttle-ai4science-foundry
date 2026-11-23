#!/usr/bin/env bash
# Runs OpenFold monomer inference for 6KWC_1. Writes to $OUTPUT_DIR
# (default: results/run-<slurm job id>/).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ -z "${CONDA_PREFIX:-}" || "$CONDA_PREFIX" != "$OPENFOLD_ENV" ]]; then
  load_modules
  activate_env
fi

mkdir -p "$OUTPUT_DIR"
echo "output directory: $OUTPUT_DIR"

cd "$OPENFOLD_DIR"

python run_pretrained_openfold.py "$FASTA_DIR" "$TEMPLATE_MMCIF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_precomputed_alignments "$ALIGNMENT_DIR" \
  --config_preset "$CONFIG_PRESET" \
  --openfold_checkpoint_path "$CHECKPOINT_PATH" \
  --model_device "cuda:0" \
  --data_random_seed 0 \
  --skip_relaxation

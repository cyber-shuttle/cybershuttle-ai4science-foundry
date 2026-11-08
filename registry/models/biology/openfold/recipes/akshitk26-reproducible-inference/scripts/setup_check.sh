#!/usr/bin/env bash
# Checks that the env, inputs, and GPU are ready. Works with or without a .env
# file and does not require you to activate conda yourself.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ -z "${CONDA_PREFIX:-}" || "$CONDA_PREFIX" != "$OPENFOLD_ENV" ]]; then
  load_modules
  activate_env
fi

for required_path in "$OPENFOLD_DIR" "$FASTA_DIR/6KWC_1.fasta" "$ALIGNMENT_DIR/6KWC_1" "$TEMPLATE_MMCIF_DIR" "$CHECKPOINT_PATH"; do
  [[ -e "$required_path" ]] || die "missing path: $required_path (see README steps 3 and 4)"
done

python - <<'PY'
import torch
import openfold

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("cuda unavailable: run this on a GPU node (salloc or sbatch)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"openfold module: {openfold.__file__}")
PY

echo "setup check passed"

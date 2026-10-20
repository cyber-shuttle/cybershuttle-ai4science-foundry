#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "missing .env"
  exit 1
fi

# shellcheck disable=SC1091
source .env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for required_path in "$OPENFOLD_DIR" "$FASTA_DIR" "$ALIGNMENT_DIR" "$TEMPLATE_MMCIF_DIR" "$CHECKPOINT_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing path: $required_path"
    exit 1
  fi
done

python - <<'PY'
import torch
import openfold

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("cuda unavailable")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"openfold module: {openfold.__file__}")
PY

echo "setup check passed"

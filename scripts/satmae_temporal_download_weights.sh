#!/usr/bin/env bash
# Download the official temporal ViT-Large fine-tune checkpoint and record SHA-256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=satmae_temporal_paths.sh
source "$SCRIPT_DIR/satmae_temporal_paths.sh"

WEIGHTS_DIR="$SATMAE_TEMPORAL_WORKDIR/weights"
mkdir -p "$WEIGHTS_DIR"

URL="https://zenodo.org/record/7369797/files/finetune_fmow_temporal.pth"
OUT="$WEIGHTS_DIR/finetune_fmow_temporal.pth"
SIDECAR="$WEIGHTS_DIR/finetune_fmow_temporal.pth.sha256"

if [[ -f "$OUT" ]]; then
  echo "Checkpoint already present: $OUT"
else
  echo "Downloading $URL"
  wget --continue --progress=dot:giga -O "$OUT" "$URL"
fi

sha256sum "$OUT" | tee "$SIDECAR"
ls -lh "$OUT"
echo "DOI: 10.5281/zenodo.7369796"

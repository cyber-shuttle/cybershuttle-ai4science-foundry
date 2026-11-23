#!/usr/bin/env bash
# Download fMoW-Temporal evaluation inputs:
#   1. val_62classes.csv from the SatMAE metadata Google Drive folder
#   2. fMoW-RGB validation imagery from the public SpaceNet S3 prefix
#
# Train imagery is not downloaded. Full fMoW-RGB is ~200 GB; val-only is the
# intended path so the job stays inside the 300 GB ICE scratch quota.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=satmae_temporal_paths.sh
source "$SCRIPT_DIR/satmae_temporal_paths.sh"

DATA_DIR="$SATMAE_TEMPORAL_WORKDIR/data"
mkdir -p "$DATA_DIR/fmow"

METADATA_FOLDER_URL="https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD"
S3_VAL="s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/val"

echo "=== SatMAE metadata CSVs ==="
if [[ -f "$DATA_DIR/val_62classes.csv" ]]; then
  echo "val_62classes.csv already present"
else
  if command -v gdown >/dev/null 2>&1; then
    gdown --folder "$METADATA_FOLDER_URL" -O "$DATA_DIR/metadata_drive" --remaining-ok || true
    if [[ -f "$DATA_DIR/metadata_drive/val_62classes.csv" ]]; then
      cp "$DATA_DIR/metadata_drive/val_62classes.csv" "$DATA_DIR/val_62classes.csv"
    fi
    if [[ -f "$DATA_DIR/metadata_drive/train_62classes.csv" ]]; then
      cp "$DATA_DIR/metadata_drive/train_62classes.csv" "$DATA_DIR/train_62classes.csv"
    fi
  else
    echo "gdown is not on PATH. Install it in the satmae-temporal env or download the CSVs by hand from:"
    echo "  $METADATA_FOLDER_URL"
  fi
fi

if [[ ! -f "$DATA_DIR/val_62classes.csv" ]]; then
  echo "WARNING: val_62classes.csv is still missing. Place it at $DATA_DIR/val_62classes.csv before prepare/eval."
fi

echo "=== fMoW-RGB val imagery (public S3, no AWS credentials) ==="
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required. Activate the satmae-temporal env or module-load awscli." >&2
  exit 1
fi

aws s3 sync --no-sign-request "$S3_VAL" "$DATA_DIR/fmow/val"

echo "=== reconstruct SatMAE *_rgb_crop_0.jpg AOI chips ==="
python "$SCRIPT_DIR/satmae_temporal_crop_fmow.py" \
  --val-root "$DATA_DIR/fmow/val" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

echo "=== sizes ==="
du -sh "$DATA_DIR" "$DATA_DIR/fmow/val" 2>/dev/null || true
if [[ -f "$DATA_DIR/val_62classes.csv" ]]; then
  wc -l "$DATA_DIR/val_62classes.csv"
fi
echo "Imagery source: $S3_VAL"
echo "fMoW license: Functional Map of the World Challenge Public License"

#!/usr/bin/env bash
# Downloads fMoW-Sentinel from the Stanford Digital Repository
# (DOI 10.25740/vg497cb6002, https://purl.stanford.edu/vg497cb6002, Version 1)
# and extracts ONLY the validation split.
#
# Writes to $SATMAE_WORKDIR/data (default ~/scratch/satmae/data), not the
# foundry checkout: the archive alone is ~77.5 GB.
set -euo pipefail

SATMAE_WORKDIR="${SATMAE_WORKDIR:-$HOME/scratch/satmae}"
DATA_DIR="$SATMAE_WORKDIR/data"
BASE="https://stacks.stanford.edu/file/druid:vg497cb6002"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[$(date)] Downloading metadata files..."
wget --continue "$BASE/README.md" -O README_fmow_sentinel.md
wget --continue "$BASE/val.csv"   -O val.csv

echo "[$(date)] Downloading fmow-sentinel.tar.gz (77.5 GB)..."
wget --continue --progress=dot:giga "$BASE/fmow-sentinel.tar.gz" -O fmow-sentinel.tar.gz

echo "[$(date)] Checksums:"
sha256sum val.csv fmow-sentinel.tar.gz | tee checksums_data.sha256

echo "[$(date)] Archive layout (first entries):"
tar -tzf fmow-sentinel.tar.gz | head -5 || true

echo "[$(date)] Extracting validation split only..."
tar -xzf fmow-sentinel.tar.gz --wildcards '*/val/*'

echo "[$(date)] Extracted image count:"
find "$DATA_DIR" -path '*/val/*' -name '*.tif' | wc -l
du -sh "$DATA_DIR"
echo "[$(date)] Done."

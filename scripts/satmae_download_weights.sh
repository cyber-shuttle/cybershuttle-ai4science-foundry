#!/usr/bin/env bash
# Downloads the SatMAE ViT-Base multi-spectral FINE-TUNED checkpoint
# (finetune-vit-base-e7.pth) from the Zenodo record for SatMAE
# (DOI 10.5281/zenodo.7338613) and records its SHA-256 checksum.
#
# Writes to $SATMAE_WORKDIR/weights (default ~/scratch/satmae/weights).
set -euo pipefail

SATMAE_WORKDIR="${SATMAE_WORKDIR:-$HOME/scratch/satmae}"
WEIGHTS_DIR="${SATMAE_WORKDIR}/weights"
URL="https://zenodo.org/record/7338613/files/finetune-vit-base-e7.pth"
OUT="${WEIGHTS_DIR}/finetune-vit-base-e7.pth"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${OUT}" ]; then
  echo "Checkpoint already present at ${OUT}, skipping download."
else
  echo "Downloading ${URL}"
  wget --continue --tries=3 -O "${OUT}" "${URL}"
fi

echo "Computing SHA-256 checksum..."
shasum -a 256 "${OUT}" > "${OUT}.sha256" 2>/dev/null || sha256sum "${OUT}" > "${OUT}.sha256"

echo "Checksum written to ${OUT}.sha256:"
cat "${OUT}.sha256"
echo
echo "Record this checksum in registry/models/earth-science/satmae/README.md (Hardware and provenance)."

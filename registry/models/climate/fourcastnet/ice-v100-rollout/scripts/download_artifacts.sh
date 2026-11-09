#!/usr/bin/env bash
# Downloads the FourCastNet demo bundle (pretrained backbone weights, normalization
# statistics, and a small out-of-sample 2018 ERA5 evaluation file) from NERSC over
# plain HTTPS, extracts it, and verifies every file against docs/CHECKSUMS.sha256.
# No Globus account is needed for this bundle.
#
# Usage: ./scripts/download_artifacts.sh [ARTIFACT_ROOT]
#   default ARTIFACT_ROOT: $HOME/scratch/fcn-artifacts
#
# The download is resumable (wget -c). If the login node kills the transfer
# (exit code 137, see docs/TROUBLESHOOTING.md), run this inside an interactive
# `srun` session instead and re-run it to resume.

set -euo pipefail

ARTIFACT_ROOT="${1:-$HOME/scratch/fcn-artifacts}"
URL="https://portal.nersc.gov/project/m4134/ccai_demo.tar"
EXPECTED_BYTES=3998497644
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ARTIFACT_ROOT"
cd "$ARTIFACT_ROOT"

echo "Downloading (resumable) from $URL ..."
wget -c -O ccai_demo.tar "$URL"

actual=$(stat -c %s ccai_demo.tar)
if [ "$actual" -ne "$EXPECTED_BYTES" ]; then
  echo "ERROR: ccai_demo.tar is $actual bytes, expected $EXPECTED_BYTES. Re-run to resume." >&2
  exit 1
fi

echo "Extracting ..."
tar -xzf ccai_demo.tar

echo "Verifying checksums against docs/CHECKSUMS.sha256 ..."
mkdir -p "$REPO_DIR/logs"
{
  echo "# Download manifest, generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Source: $URL"
  sha256sum -c "$REPO_DIR/docs/CHECKSUMS.sha256"
} | tee "$REPO_DIR/logs/download_manifest.txt"

echo "Done. Artifacts are under $ARTIFACT_ROOT/ccai_demo"

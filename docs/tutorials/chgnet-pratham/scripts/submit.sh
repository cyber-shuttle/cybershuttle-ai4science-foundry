#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MAKERSPACE_PARTITION:?Set MAKERSPACE_PARTITION to your course-authorized partition}"
: "${MAKERSPACE_ACCOUNT:?Set MAKERSPACE_ACCOUNT to your course-authorized account}"
mkdir -p runs
options=(--partition="$MAKERSPACE_PARTITION" --account="$MAKERSPACE_ACCOUNT")
if [[ -n "${MAKERSPACE_QOS:-}" ]]; then options+=(--qos="$MAKERSPACE_QOS"); fi
sbatch "${options[@]}" scripts/inference.sbatch

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
run_id=${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir="runs/$run_id"
mkdir -p "$run_dir"
# Preserve full terminal output and a failure exit status.
exec > >(tee "$run_dir/execution.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-status.txt"' EXIT
printf 'Run ID: %s\nUTC start: %s\n' "$run_id" "$(date -u +%FT%TZ)"
hostname
python --version
python -m pip check
python -m pip freeze > "$run_dir/environment-lock.txt"
cp config/workflow.json "$run_dir/"
cp artifacts/manifest.json "$run_dir/"
if command -v nvidia-smi >/dev/null; then nvidia-smi > "$run_dir/nvidia-smi.txt"; fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-job.txt"; fi
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python scripts/infer.py --device "${DEVICE:-cuda}" --output "$run_dir/results.json"
printf 'UTC finish: %s\n' "$(date -u +%FT%TZ)"

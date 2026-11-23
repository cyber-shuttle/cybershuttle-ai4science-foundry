#!/usr/bin/env bash
# Proof-of-concept batch job. Submit from the recipe directory:
#   sbatch poc/poc.sbatch
# Run python poc/fetch_inputs.py on a login node first (it needs internet).
# Same account and GPU notes as jobs/openfold_inference.sbatch.
#SBATCH --job-name=openfold-poc
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=rtx_6000:1
#SBATCH --time=01:30:00
#SBATCH --output=poc/evidence/slurm-%j.out
#SBATCH --error=poc/evidence/slurm-%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
[[ -f scripts/common.sh ]] || { echo "submit this job from the recipe directory" >&2; exit 1; }
source scripts/common.sh
command -v nvidia-smi >/dev/null 2>&1 \
  || die "no NVIDIA GPU on $(hostname); request an NVIDIA type such as --gpus=rtx_6000:1"
load_modules
activate_env

{
  date
  hostname
  module list
  nvidia-smi
} > "poc/evidence/platform-${SLURM_JOB_ID}.txt" 2>&1

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total \
  --format=csv,noheader,nounits --loop=1 \
  > "poc/evidence/gpu-memory-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

bash scripts/setup_check.sh
/usr/bin/time -v bash poc/run_poc.sh
date

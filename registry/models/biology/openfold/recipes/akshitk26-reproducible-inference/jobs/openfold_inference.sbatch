#!/usr/bin/env bash
# Submit from the recipe directory:
#   sbatch jobs/openfold_inference.sbatch
# If your charge account is not "coc", override it on the command line:
#   sbatch --account=<your-account> jobs/openfold_inference.sbatch
# (list yours with: sacctmgr -nP show assoc where user=$USER format=account)
# The GPU type is pinned because ice-gpu also has AMD GPUs (mi210, r9700),
# which cannot run OpenFold's CUDA code. rtx_6000 is the validated type; any
# NVIDIA type works, e.g. sbatch --gpus=v100:1 jobs/openfold_inference.sbatch
# (list types with: sinfo -p ice-gpu -h -o "%G" | sort -u)
#SBATCH --job-name=openfold-6KWC1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=rtx_6000:1
#SBATCH --time=02:00:00
#SBATCH --output=evidence/slurm-%j.out
#SBATCH --error=evidence/slurm-%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
[[ -f scripts/common.sh ]] || { echo "submit this job from the recipe directory" >&2; exit 1; }
source scripts/common.sh
command -v nvidia-smi >/dev/null 2>&1 \
  || die "no NVIDIA GPU on $(hostname); request an NVIDIA type such as --gpus=rtx_6000:1"
load_modules
activate_env

pred="$OUTPUT_DIR/predictions/6KWC_1_${CONFIG_PRESET}_unrelaxed.pdb"

mkdir -p evidence
{
  date
  hostname
  module list
  nvidia-smi
} > "evidence/platform-${SLURM_JOB_ID}.txt" 2>&1

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total \
  --format=csv,noheader,nounits --loop=1 \
  > "evidence/gpu-memory-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

bash scripts/setup_check.sh
{ /usr/bin/time -v bash scripts/run_inference.sh; } 2>&1 | tee "evidence/inference-${SLURM_JOB_ID}.log"

python scripts/validate_output.py "$pred" | tee "evidence/validation-${SLURM_JOB_ID}.log"
python scripts/compare_ca_rmsd.py "$pred" A inputs/reference/6kwc.pdb A \
  | tee "evidence/structure-comparison-${SLURM_JOB_ID}.log"

echo "prediction: $pred"
date

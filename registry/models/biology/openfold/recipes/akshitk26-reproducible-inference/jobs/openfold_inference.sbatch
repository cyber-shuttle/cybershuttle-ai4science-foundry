#!/usr/bin/env bash
#SBATCH --job-name=openfold-6KWC1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=evidence/slurm-%j.out
#SBATCH --error=evidence/slurm-%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?}"

module purge
module load gcc/12.3.0
module load cuda/12.1.1
module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openfold-repro
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

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
python scripts/validate_output.py \
  results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb \
  | tee "evidence/validation-${SLURM_JOB_ID}.log"

date

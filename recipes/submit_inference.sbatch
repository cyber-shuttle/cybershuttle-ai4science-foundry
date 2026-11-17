#!/bin/bash
#SBATCH --job-name=fno_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_fno_inference_%j.out

set -euo pipefail

echo "=========================================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Compute Node: $(hostname)"
echo "Submit Time:  $(date)"
echo "=========================================================="
echo "Allocated GPU Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

module purge
module load anaconda3

# Explicitly initialize Conda for non-interactive Slurm batch shells
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate environment (supports scratch prefix or named environment)
if [[ -d "${HOME}/scratch/conda-envs/fno-inference" ]]; then
    conda activate "${HOME}/scratch/conda-envs/fno-inference"
else
    conda activate fno-inference
fi

if [[ -f "run_inference.py" ]]; then
    python run_inference.py
else
    python scripts/run_inference.py
fi

echo "=========================================================="
echo "Job Completion Time: $(date)"
echo "=========================================================="

#!/bin/bash
#SBATCH --job-name=dimenet_inference
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1               
#SBATCH --time=00:30:00            
#SBATCH --output=execution_%j.log  

# ==============================================================================
# Portability Note for External Users:
# If your PACE ICE allocation is not 'coc', update --account and --partition above
# or pass overrides via CLI: sbatch --account=<account> --partition=<partition> run_inference.slurm
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting DimeNet++ Job on PACE ICE"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

# Protect user home directory quota by directing caches to scratch storage
export TORCH_HOME="${HOME}/scratch/.cache/torch"
export PIP_CACHE_DIR="${HOME}/scratch/.cache/pip"
mkdir -p "$TORCH_HOME" "$PIP_CACHE_DIR"

# Load necessary modules for GT PACE ICE
module load anaconda3

# Activate the conda environment
source activate dimenet-env

# 1. Run the Smoke Test Validation
echo "--- RUNNING SMOKE TEST ---"
python evaluate.py --smoke-test

# 2. Run the Full Inference Validation
echo "--- RUNNING FULL INFERENCE ---"
python evaluate.py --target 7

echo "Job completed successfully."
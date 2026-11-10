#!/usr/bin/env bash
#SBATCH --job-name=chgnet-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
set -euo pipefail
# Submit from the project root. Partition/account/QoS are supplied at submission.
cd "$SLURM_SUBMIT_DIR"
# Baseline module on ICE; verify availability in your Makerspace allocation.
module load anaconda3/2023.03
export DEVICE=cuda
bash scripts/run.sh

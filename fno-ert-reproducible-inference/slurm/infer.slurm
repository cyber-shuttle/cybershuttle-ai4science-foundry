#!/bin/bash
#SBATCH -J fno-infer
#SBATCH -A coc
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err

cd $SLURM_SUBMIT_DIR
PY=/storage/ice1/0/5/ityagi8/envs/fno-ert/bin/python
nvidia-smi
echo "===== SMOKE TEST ====="; $PY -u scripts/smoke_test.py
echo "===== INFERENCE =====";  $PY -u scripts/infer.py
echo "===== VALIDATION =====";  $PY -u scripts/validate.py

#!/bin/bash
#SBATCH -J fno-train
#SBATCH -A coc
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

cd $SLURM_SUBMIT_DIR
PY=/storage/ice1/0/5/ityagi8/envs/fno-ert/bin/python
nvidia-smi
$PY -u scripts/train.py --config configs/fno_ert.yaml

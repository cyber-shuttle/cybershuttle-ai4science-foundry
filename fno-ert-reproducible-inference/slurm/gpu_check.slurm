#!/bin/bash
#SBATCH -J gpu-check
#SBATCH -A coc
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o logs/gpu_check_%j.out
#SBATCH -e logs/gpu_check_%j.err

cd $SLURM_SUBMIT_DIR
PY=/storage/ice1/0/5/ityagi8/envs/fno-ert/bin/python
nvidia-smi
$PY -c "import torch, neuralop; print(torch.__version__, torch.cuda.is_available(), neuralop.__version__)"

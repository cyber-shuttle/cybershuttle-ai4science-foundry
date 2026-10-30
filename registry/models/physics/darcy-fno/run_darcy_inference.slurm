#!/bin/bash
#SBATCH -A coc
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o darcy_inference_%j.out

module load anaconda3
conda activate neuralop

python darcy_inference.py

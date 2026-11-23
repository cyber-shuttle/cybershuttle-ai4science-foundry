#!/bin/bash
#SBATCH -J train100
#SBATCH -A coc
#SBATCH -q coc-ice
#SBATCH -p ice-gpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/train100-%j.out

cd $SLURM_SUBMIT_DIR
source ./config.sh
require_data

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
which python

hostname
nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

cd Car-Design-ShapeNetCar
python -u main.py \
  --cfd_model=Transolver \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --nb_epochs 100 \
  --preprocessed 1

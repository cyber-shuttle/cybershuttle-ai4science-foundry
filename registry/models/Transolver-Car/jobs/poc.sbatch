#!/bin/bash
#SBATCH -A coc
#SBATCH -q coc-ice
#SBATCH -p ice-gpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 3:00:00
# Job name and log file are set at submission time (see submit_poc.sh):
#   sbatch -J poc-hybrid200 -o logs/poc-hybrid200-%j.out jobs/poc.sbatch --model hybrid --enc_epochs 200 --drag

cd $SLURM_SUBMIT_DIR
source ./config.sh

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
which python

# utils/drag_coefficient.py reads these (same as the eval jobs)
export MLCFD_DATA_DIR="$DATA_DIR"
export MLCFD_SAVE_DIR="$SAVE_DIR"

hostname
nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

cd Car-Design-ShapeNetCar
python -u hybrid_deeponet.py \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  "$@"

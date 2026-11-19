#!/bin/bash
#SBATCH -J satmae-build-env
#SBATCH -A coc
#SBATCH -p coc-cpu
#SBATCH --qos=coc-ice
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Builds the SatMAE conda env under ~/scratch (ICE home quota is 30 GB) and
# clones upstream SatMAE at the pinned commit. Submit from the foundry repo
# root: sbatch scripts/satmae_build_env.sbatch
# Afterwards, run scripts/satmae_patch_timm.sh once inside the env.

SATMAE_WORKDIR="${SATMAE_WORKDIR:-$HOME/scratch/satmae}"
SATMAE_REPO="${SATMAE_REPO:-$SATMAE_WORKDIR/SatMAE}"
SATMAE_COMMIT=0b210aceb37a14bbbd897110db5b104b3271d818
mkdir -p "$SATMAE_WORKDIR/env"

if [ ! -d "$SATMAE_REPO/.git" ]; then
  git clone https://github.com/sustainlab-group/SatMAE "$SATMAE_REPO"
fi
git -C "$SATMAE_REPO" checkout "$SATMAE_COMMIT"

module load anaconda3
export CONDA_PKGS_DIRS=$HOME/scratch/conda_pkgs
conda env create -f scripts/satmae_environment.yml -p $HOME/scratch/envs/satmae
source activate $HOME/scratch/envs/satmae
python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'timm', timm.__version__)"
conda env export -p $HOME/scratch/envs/satmae > "$SATMAE_WORKDIR/env/environment.lock.yml"
pip freeze > "$SATMAE_WORKDIR/env/pip-freeze.txt"

#!/bin/bash
# One-time setup. Run on the LOGIN node (needs internet):  bash setup_env.sh
# Creates the "cebra" conda env, installs packages, records versions, downloads the data.
set -euo pipefail

# run from the folder this script is in, no matter where it was called from
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Project folder: $PWD"
mkdir -p data env logs outputs   # logs/ must exist before sbatch, or the job dies with no log

ENV_NAME=${ENV_NAME:-cebra}
PY_VER=${PY_VER:-3.11}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}

module load anaconda3 2>/dev/null || module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || \
  echo "WARN: no conda module loaded; assuming conda is already on PATH"

# keep caches in scratch, not home (small quota)
SCRATCH_DIR=${SCRATCH_DIR:-${SCRATCH:-$PWD}}
export CONDA_PKGS_DIRS="$SCRATCH_DIR/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_DIR/.pip_cache"

# making the environment directory 
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# the packages and installs
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX" #specific torch version for the CUDA
pip install -r requirements.txt

# exact environment for tracibility
pip freeze > env/pip_freeze.txt
conda env export --no-builds > env/environment_export.yml
python - <<'EOF' | tee env/versions.txt
import sys, torch, cebra, numpy, sklearn
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("cebra", cebra.__version__)
print("numpy", numpy.__version__, "sklearn", sklearn.__version__)
EOF

# data downloading
export CEBRA_DATADIR="$PWD/data"
python download_data.py | tee env/data_manifest.txt

echo "Setup complete. Next: run sbatch run_cebra.sbatch to start!!"

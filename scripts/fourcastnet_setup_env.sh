#!/bin/bash
# Build the conda env and check out the NVlabs code. Run on the login node from the repo root.
set -eo pipefail

source scripts/fourcastnet_config.env
module load $MODULES
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$WORK_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$EVIDENCE_DIR/env"

# Conda env (in scratch)
if [ ! -d "$CONDA_ENV_PREFIX" ]; then
    conda create -y -p "$CONDA_ENV_PREFIX" python="$PYTHON_VERSION" pip
fi
conda activate "$CONDA_ENV_PREFIX"
pip install -r scripts/fourcastnet_requirements.txt

# NVlabs FourCastNet at the pinned commit
if [ ! -d "$FCN_REPO_DIR/.git" ]; then
    git clone "$FCN_REPO_URL" "$FCN_REPO_DIR"
fi
git -C "$FCN_REPO_DIR" fetch --quiet origin
git -C "$FCN_REPO_DIR" checkout --quiet "$FCN_REPO_COMMIT"

# Check the model code imports with these versions
(cd "$FCN_REPO_DIR" && python -c "from networks.afnonet import AFNONet; print('AFNONet import ok')")

# Save exact versions for reproducibility
pip freeze > "$EVIDENCE_DIR/env/pip-freeze.txt"
conda env export -p "$CONDA_ENV_PREFIX" > "$EVIDENCE_DIR/env/conda-env.yml"
{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "repo commit: $(git rev-parse HEAD)"
    echo "FourCastNet commit: $(git -C "$FCN_REPO_DIR" rev-parse HEAD)"
    echo "module: $MODULES"
    python -c "import sys, torch, numpy; print('python:', sys.version.split()[0]); print('torch:', torch.__version__, 'cuda', torch.version.cuda); print('numpy:', numpy.__version__)"
} | tee "$EVIDENCE_DIR/env/versions.txt"

echo "Setup done. Env: $CONDA_ENV_PREFIX"

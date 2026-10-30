#!/bin/bash
# Set up a HyenaDNA conda environment, picking a PyTorch build that matches
# the GPU actually allocated to this job. Run on a GPU node (compute
# capability detection needs nvidia-smi to see a real GPU) — on PACE ICE
# that means inside an `srun`/`sbatch` allocation on the ice-gpu partition,
# not on the login node.
#
# Must be sourced, not executed: `conda activate` and the cache-redirect
# exports below only take effect in the shell that runs them, and would be
# silently lost on return if this ran in its own subshell.
#
# Usage: source ./hyenadna_setup_env.sh [env_prefix] [repo_dir]
(return 0 2>/dev/null) || {
    echo "hyenadna_setup_env.sh must be sourced, not executed directly." >&2
    echo "Run: source ${BASH_SOURCE[0]} [env_prefix] [repo_dir]" >&2
    exit 1
}
set -euo pipefail

ENV_PREFIX="${1:-$HOME/scratch/.conda/envs/hyena-dna}"
REPO_DIR="${2:-$HOME/scratch/hyena-dna}"

# PACE ICE home quota (~30GB) is consumed fast by package caches long before
# scratch fills up. Every cache that defaults under $HOME needs redirecting.
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export GIT_CONFIG_GLOBAL="$HOME/scratch/.gitconfig"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

if [ ! -d "$REPO_DIR" ]; then
    git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git "$REPO_DIR"
fi

# purge first: a fresh shell/job starts with anaconda3 loaded by default,
# and loading miniforge on top triggers Lmod's automatic swap-out of
# anaconda3, whose deactivation hook references an unset variable -- fatal
# under `set -u` (confirmed live: killed an sbatch job in 11s before it ever
# reached this script). Doing it here, not just in the callers, covers the
# README's own "source this directly in a fresh srun shell" path too.
module purge
module load miniforge
mamba create -y -p "$ENV_PREFIX" python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Pick a PyTorch build that actually has compiled kernels for the allocated
# GPU. The repo's documented pin, torch==1.13.0+cu117, only ships kernels
# through compute capability 8.6 (Ampere); it silently no-ops ("no kernel
# image is available") on Hopper (compute capability 9.0: H100/H200), which
# PACE ICE's ice-gpu partition can also allocate. Installing via conda also
# hits an unrelated MKL/ITT symbol mismatch on this cluster's image, so both
# branches use pip wheels from the official PyTorch index, not conda.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
echo "Detected GPU compute capability: $COMPUTE_CAP"

if awk "BEGIN{exit !($COMPUTE_CAP >= 9.0)}"; then
    echo "Hopper or newer -> torch 2.1.2+cu118"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "Volta/Ampere -> torch 1.13.0+cu117 (repo's documented pin)"
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
fi

# requirements.txt pulls in torchtext==0.14.0, which hard-pins torch==1.13.0
# and silently downgrades whichever GPU build was just installed (pip warns
# about the conflict but does not fail). Reinstalling afterward, not before,
# is what actually matters here.
pip install -r "$REPO_DIR/requirements.txt"
if awk "BEGIN{exit !($COMPUTE_CAP >= 9.0)}"; then
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118
fi

# flash_attn is a hard import in src/models/sequence/dna_embedding.py (the
# Lightning training path), but standalone_hyenadna.py -- what
# scripts/hyenadna_inference.py and hyenadna_validate.py actually use --
# imports nothing from it. The pip wheel is not a prebuilt binary, so
# installing it means a lengthy from-source compile; skip it by default and
# only pay that cost for the training path.
if [ "${HYENADNA_INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    pip install flash-attn --no-build-isolation
else
    echo "Skipping flash-attn (not needed for standalone inference)." \
         "Set HYENADNA_INSTALL_FLASH_ATTN=1 before running this script if you need the Lightning training path."
fi

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "Environment ready at $ENV_PREFIX"

#!/bin/bash
#SBATCH --job-name=hyenadna-train
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hyenadna-train-%j.out
#
# Reproducible from-scratch HyenaDNA training job for PACE ICE (ice-gpu
# partition): the repo's own "quick entry point" -- a small 2-layer model
# trained on one GenomicBenchmarks task -- not full hg38 pretraining (see
# the README's Training section for why that's a separate, heavier thing).
#
# Usage: sbatch hyenadna_train.sbatch [experiment] [hydra_run_dir]

set -euo pipefail
mkdir -p logs

REPO_DIR="$HOME/scratch/hyena-dna"
ENV_PREFIX="$HOME/scratch/.conda/envs/hyena-dna"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-hg38/genomic_benchmark_scratch}"
RUN_DIR="${2:-$REPO_DIR/hyena_train_runs/${SLURM_JOB_ID}}"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    HYENADNA_INSTALL_FLASH_ATTN=1 source "$SCRIPT_DIR/hyenadna_setup_env.sh" "$ENV_PREFIX" "$REPO_DIR"
else
    conda activate "$ENV_PREFIX"
fi

# The env may already exist from an inference-only setup, which skips
# flash-attn (dna_embedding.py -- the module train.py's Lightning path
# always loads -- hard-imports it; there's no pure-PyTorch fallback here,
# unlike the standalone inference path).
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn missing from existing env; installing (this compiles from source)..."
    export PIP_CACHE_DIR="$HOME/scratch/.cache/pip" TMPDIR="$HOME/scratch/tmp"
    pip install flash-attn --no-build-isolation
fi

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

# model.fused_dropout_add_ln / model.fused_mlp: the experiment config
# defaults both to values that need dropout_add_layer_norm, an optional
# fused CUDA kernel from flash-attn's csrc/layer_norm that this setup does
# not build (it's a speed optimization, not required for correctness) --
# left enabled, train.py fails with "dropout_add_layer_norm is not
# installed" before a single batch runs.
python -m train wandb=null experiment="$EXPERIMENT" \
    model.fused_dropout_add_ln=false model.fused_mlp=false \
    hydra.run.dir="$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

python "$SCRIPT_DIR/hyenadna_train_validate.py" \
    --log-file "$RUN_DIR/train.log" \
    --checkpoint-dir "$RUN_DIR/checkpoints"

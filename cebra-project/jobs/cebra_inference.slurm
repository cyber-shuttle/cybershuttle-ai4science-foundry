#!/bin/bash
#SBATCH --job-name=cebra-hippocampus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p coc-gpu
#SBATCH -q coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cebra-%j.out
#SBATCH --error=logs/cebra-%j.err

# Tested resource configuration; environment and storage paths are configurable.

set -euo pipefail
trap 'status=$?; echo "Job exit status: ${status}; finished: $(date -u +%FT%TZ)"; exit "$status"' EXIT

: "${SLURM_JOB_ID:?Submit with sbatch or run inside a SLURM allocation.}"
: "${SLURM_SUBMIT_DIR:?Submit from the project root.}"
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f run_inference.py || ! -d logs ]]; then
    echo "Submit from the project root after creating logs/." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
: "${CEBRA_ENV:?Set CEBRA_ENV to the path of your CEBRA environment before submitting.}"
if [[ -d "$CEBRA_ENV/conda-meta" ]]; then
    : "${CONDA_EXE:?Initialize and activate Conda before submitting this job.}"
    cebra_conda_hook=$("$CONDA_EXE" shell.bash hook)
    eval "$cebra_conda_hook"
    conda activate "$CEBRA_ENV"
else
    source "$CEBRA_ENV/bin/activate"
fi
python - <<'PY'
import os
from pathlib import Path
import sys
import pip

expected = Path(os.environ["CEBRA_ENV"]).resolve()
if Path(sys.prefix).resolve() != expected or not Path(pip.__file__).resolve().is_relative_to(expected):
    raise SystemExit("Python and pip must come from CEBRA_ENV.")
PY
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date -u +%FT%TZ)"
python --version
python -m pip check

# Each job keeps its own checkpoint, embeddings and execution evidence.
run_dir="results/job-${SLURM_JOB_ID}"
python -u run_inference.py --seed 42 --data-dir "${CEBRA_WORKDIR:-.}/data" --output-dir "$run_dir"
python -u validation/validate_results.py --results-dir "$run_dir"

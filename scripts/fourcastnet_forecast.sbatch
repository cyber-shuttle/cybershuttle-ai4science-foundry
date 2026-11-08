#!/bin/bash
#SBATCH --job-name=fourcastnet
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=registry/models/climate/fourcastnet/evidence/%x-%j.out
# Partition, account, QOS and GPU come from the SBATCH_* variables in the config.
# Usage, from the repo root:  source scripts/fourcastnet_config.env && sbatch scripts/fourcastnet_forecast.sbatch [nsteps]
set -eo pipefail

source scripts/fourcastnet_config.env
module load $MODULES
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"

NSTEPS=${1:-$NSTEPS}
RUN_DIR="$WORK_DIR/runs/$SLURM_JOB_ID"
OUT_DIR="$EVIDENCE_DIR/$SLURM_JOB_ID"

echo "date: $(date -Is)"
echo "host: $(hostname)"
echo "job: $SLURM_JOB_ID  partition: $SLURM_JOB_PARTITION  nsteps: $NSTEPS  init times: $INIT_TIMES"
echo "repo commit: $(git rev-parse HEAD)"
echo "FourCastNet commit: $(git -C "$FCN_REPO_DIR" rev-parse HEAD)"
module list 2>&1
python -c "import sys, torch, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', numpy.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SECONDS=0
python scripts/fourcastnet_inference.py --nsteps "$NSTEPS" --out "$RUN_DIR"
echo "forecast time: ${SECONDS}s"

python scripts/fourcastnet_validate.py "$RUN_DIR" --out "$OUT_DIR"

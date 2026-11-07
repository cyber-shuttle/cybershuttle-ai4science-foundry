#!/usr/bin/env bash
#SBATCH --job-name=chgnet-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/chgnet-%j.out
#SBATCH --error=logs/chgnet-%j.err

# ICE runs Slurm as of 2026. Select the correct course QOS at submission:
#   sbatch --qos=pace-ice jobs/ice_gpu.sbatch smoke
#   OR: sbatch --qos=coc-ice jobs/ice_gpu.sbatch smoke
# Adjust QOS, account, partition, GRES to what your ICE access permits.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

module load anaconda3
# Initialize conda reliably in a noninteractive scheduler shell.
eval "$(conda shell.bash hook)"
conda activate chgnet-repro

MODE="${1:-smoke}"
OUT="results/slurm-${SLURM_JOB_ID}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf 'UTC START: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'NODE: %s | JOB: %s | MODE: %s\n' "$(hostname)" "$SLURM_JOB_ID" "$MODE"
python scripts/capture_environment.py --out-dir "$OUT"

case "$MODE" in
  smoke)
    srun python scripts/run_inference.py smoke --device cuda --out-dir "$OUT"
    (
      cd "$OUT"
      sha256sum run_metadata.json smoke_predictions.json smoke_validation.json \
        environment.json pip_freeze.txt nvidia_smi.txt slurm_job_info.txt \
        > artifact_checksums.sha256
    )
    ;;
  benchmark)
    SUBSET="${2:-inputs/mptrj_subset.json}"
    srun python scripts/run_inference.py benchmark --device cuda --subset "$SUBSET" --out-dir "$OUT"
    (
      cd "$OUT"
      sha256sum run_metadata.json benchmark_predictions.json benchmark_metrics.json \
        environment.json pip_freeze.txt nvidia_smi.txt slurm_job_info.txt \
        > artifact_checksums.sha256
    )
    ;;
  *)
    echo "Mode must be smoke or benchmark" >&2
    exit 2
    ;;
esac
printf 'UTC END: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

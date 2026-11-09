#!/bin/bash
# One command for the whole Clay workflow on PACE ICE. Run on a login node
# from the Clay model folder:
#
#   bash scripts/clay_run_all.sh
#
# It runs the preflight checks, then submits the four jobs chained with
# Slurm dependencies, so each one starts only if the previous one succeeded:
#   1. clay_build_env.sbatch   (CPU)  conda env + exact package versions
#   2. clay_download.sbatch    (CPU)  checkpoint, teacher, NAIP tile, reference
#   3. clay_smoke_test.sbatch  (GPU)  8 chips: load, shape, finite, repeatable
#   4. clay_inference.sbatch   (GPU)  725 chips vs Clay's published embeddings
# Then it returns; the jobs run on their own (about 13 minutes of job time,
# plus any wait in the queue). Watch them with: squeue -u $USER
#
# To start completely fresh (new env and downloads), pick a new work folder:
#   CLAY_WORKDIR=$HOME/scratch/clay-fresh bash scripts/clay_run_all.sh
# Different Slurm account, QOS, partitions or GPU: see clay_sbatch_opts in
# scripts/clay_common.sh (SBATCH_ACCOUNT, SBATCH_QOS, CLAY_CPU_PARTITION,
# CLAY_GPU_PARTITION, CLAY_GPU_GRES).
# The job IDs are saved to benchmarks/logs/run_all-<time>.txt.

set -euo pipefail
# shellcheck source=scripts/clay_common.sh
source scripts/clay_common.sh
clay_require_repo_root

if ! bash scripts/clay_preflight.sh; then
    echo "Preflight failed; nothing was submitted." >&2
    exit 1
fi
export CLAY_WORKDIR   # sbatch passes the environment to the jobs

submit() {  # submit <job script> [extra sbatch options...]; prints the job ID
    local script="$1"; shift
    clay_sbatch_opts "$script"
    sbatch --parsable ${CLAY_SBATCH_OPTS[@]+"${CLAY_SBATCH_OPTS[@]}"} "$@" "$script"
}

build=$(submit scripts/clay_build_env.sbatch)
download=$(submit scripts/clay_download.sbatch --dependency=afterok:"$build")
smoke=$(submit scripts/clay_smoke_test.sbatch --dependency=afterok:"$download")
inference=$(submit scripts/clay_inference.sbatch --dependency=afterok:"$smoke")

record="$CLAY_EVIDENCE/logs/run_all-$(date +%Y%m%dT%H%M%S).txt"
{
    echo "submitted $(date -Is) by $USER from $(hostname)"
    clay_code_version
    echo "CLAY_WORKDIR $CLAY_WORKDIR"
    echo "slurm account ${SBATCH_ACCOUNT:-from scripts}, qos ${SBATCH_QOS:-from scripts}," \
         "cpu partition $(clay_job_partition scripts/clay_build_env.sbatch)," \
         "gpu partition $(clay_job_partition scripts/clay_inference.sbatch), gpu ${CLAY_GPU_GRES:-from scripts}"
    echo "build=$build download=$download smoke=$smoke inference=$inference"
} | tee "$record"
echo
echo "Watch:   squeue -u \$USER"
echo "Logs:    $CLAY_EVIDENCE/logs/clay-*-<jobid>.out"
echo "Result:  $CLAY_EVIDENCE/outputs/inference/summary.txt"

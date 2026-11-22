#!/bin/bash
# Run on the login node AFTER a job finishes, to save final Slurm accounting into the run folder.
#     bash collect_evidence.sh <JOB_ID>

set -euo pipefail
JOB=${1:?usage: bash collect_evidence.sh <JOB_ID>}
OUT=$(ls -d outputs/*_"$JOB" 2>/dev/null | head -1)
OUT=${OUT:-outputs/job_$JOB}; mkdir -p "$OUT"
{
  echo "# sacct"; sacct -j "$JOB" --format=JobID,JobName,Partition,AllocTRES%45,Elapsed,TotalCPU,MaxRSS,State,ExitCode
  echo; echo "# seff"; seff "$JOB" 2>/dev/null || echo "seff not available"
} | tee "$OUT/slurm_accounting.txt"
cp logs/*-"$JOB".out logs/*-"$JOB".err "$OUT/" 2>/dev/null || true
echo

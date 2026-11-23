#!/bin/bash
# Submit the whole proof of concept as ONE job (data loaded once, 7 configurations run in priority order).
# Run from the repo root: bash jobs/submit_poc.sh
mkdir -p logs
sbatch -J poc-suite -o logs/poc-suite-%j.out jobs/poc.sbatch --suite --epochs 40 --drag --cpu_time

# Troubleshooting Log

## Issue 1 — Invalid account/partition combination
Error:
`sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified`

Cause:
My SLURM account (`coc`) had no GPU partition associations, so specifying `-q gpu` or `-p pace-cpu` caused SLURM to reject the job.

Fix:
Removed the partition line entirely and allowed SLURM to auto-select the correct CPU partition:
`Partition: coc-cpu`

## Issue 2 — Invalid QoS specification
Error:
`sbatch: error: Batch job submission failed: Invalid qos specification`

Cause:
Using `-q pace-cpu` incorrectly interpreted the value as a QoS, which does not exist.

Fix:
Removed the `-q` line entirely.

## Issue 3 — Conda warning in SLURM prolog
Warning:
`CondaError: Run 'conda init' before 'conda deactivate'`

Cause:
PACE prolog scripts attempt to deactivate Conda before the job starts.

Fix:
No action needed. The warning does not affect environment activation or inference.

## Final Result
After removing partition/QoS lines, the job ran successfully on `coc-cpu` and produced valid inference output.

# Troubleshooting notes

These are the issues I expected or ran into while making the ICE workflow.

| Symptom | Likely cause | What to do |
|---|---|---|
| `sbatch: Invalid qos specification` or pending forever | Wrong QoS for your account | Check your association with `sacctmgr` as shown in README Section 3, then set `ICE_QOS` to an allowed value. My run used `coc-ice`. If the command is unavailable, check course/PACE instructions. |
| `sbatch: invalid account/partition` | Account or partition does not match your allocation | Use `sinfo -o '%P %G %a %l'` and course docs to find the allowed settings. |
| `module: command not found` / `anaconda3` unavailable | Module setup differs on the cluster | Run `module avail anaconda3` and update the job script if the module name changed. |
| `conda activate` fails | `conda` not initialized in noninteractive shell | After `module load anaconda3`, use `eval "$(conda shell.bash hook)"`, then activate `chgnet-repro`. Export the same `CONDA_ENVS_PATH` used during installation so the batch job can find it. |
| `CUDA requested but unavailable` | No GPU assigned, or CPU-only PyTorch | Check `nvidia-smi` and `torch.cuda.is_available()` inside the Slurm job. |
| CUDA out-of-memory | GPU too small or input too large | This workflow predicts one structure at a time. If needed, lower the max atom count and document the new subset settings. |
| Package install conflict | Wheel/version mismatch | Recreate from `environment.yml`, run `pip check`, and save the final `pip_freeze.txt`. |
| Checkpoint missing | Wrong CHGNet package/version | Use `chgnet==0.4.2` and verify the `pretrained/0.3.0` checkpoint file. |
| Download fails | Figshare or cluster internet issue | Download Figshare version 2 elsewhere, copy to scratch, and record the SHA-256. |
| `no space left on device` | Environment, caches, or archive filled home | The environment needs about 5 GB. Follow README Section 4 to put environments, Conda/pip caches, and temporary files on scratch before installing. Keep the archive on scratch too. |
| MPtrj selection too slow | Large JSON scan | Run it as a CPU job, not as repeated GPU-job setup. |
| Magnetic moments missing | Some labels are `None` | Skip moment MAE for that structure. Do not treat missing as zero. |
| Stress error is huge | Wrong stress sign/unit conversion | Use `-0.1 * stress_kbar` before comparing to CHGNet GPa. |
| Smoke finite-difference fails | Unit conversion or numerical issue | Check `E_total = N * E_per_atom`, `h = 0.005 A`, and preserve the failed values. |
| Benchmark looks better than paper | The subset is not the paper's test split | Do not claim this is the original held-out test set. |
| Subset SHA-256 differs from my run report | The JSON includes download time and local path | Follow README Step B to compare the records checksum and source checksum. Keep a separate full-file checksum for your own subset. |
| Verifier says `slurm_accounting.txt` is missing | The batch script does not create it | After the job finishes, run the README's `sacct` command to save it in the run directory, then run the verifier. Check the recorded state and exit code yourself. |

## Cleanup

1. Keep the real logs and validation results needed for grading.
2. Stop unnecessary jobs: `scancel JOBID`.
3. Remove the large MPtrj archive from scratch only after the source URL and checksum are documented.
4. Deactivate conda and avoid leaving unused GPU sessions running.

If a run fails, keep the stderr/logs and write down what changed before rerunning.

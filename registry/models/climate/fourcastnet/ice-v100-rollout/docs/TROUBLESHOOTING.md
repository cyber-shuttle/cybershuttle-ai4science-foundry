# Troubleshooting

## Problems I hit while building this (2026-10-12)

| What happened | How I found the cause | Fix |
|---|---|---|
| `module load miniforge` printed a conda `ParserError` pointing at `~/.condarc` line 2 | An earlier command had appended invalid YAML to the file | I backed it up and rewrote it with `>` instead of `>>`, as one valid `envs_dirs:` block |
| `wget` of `ccai_demo.tar` died at about 95% with `Exit 137` | The login node kills long, heavy processes with SIGKILL | I resumed with `wget -c`. For repeat failures, run the download inside an `srun` session |
| My home directory hit 97% of its 30 GB quota during the environment build | The quota output showed pip and conda caches in home | `pip cache purge` and `conda clean --all -y` freed about 7 GB, and the environment and artifacts live on scratch |
| `tar -tv ccai_demo.tar` failed | The archive is gzip-compressed | Use `tar -tzf` and `tar -xzf` |
| My first plan assumed Globus transfers and a different bundle layout | Listing the NERSC portal and the tar showed the real `ccai_demo/` layout and that plain HTTPS works | I rewrote `download_artifacts.sh` and every path, and added checksums |
| The default config would run off the end of the data | I measured `2018.h5` at 40 steps while `AFNO.yaml` defaults to 41 | `configure_paths.py` sets `prediction_length` from the data shape |

I caught three job-script pitfalls in review before the first submission and fixed them in `slurm/run_inference.sbatch`. Inside a Slurm job, `$0` points at a spool copy, so paths come from `$SLURM_SUBMIT_DIR`. A relative `#SBATCH --output` path resolves against the submit directory, so I submit from the repo root. And `set -u` breaks `conda activate`, so the script leaves it off.

The first real run (job 6205007, 1 minute 35 seconds, completed) taught me three more things:

- **GPU memory.** Peak GPU memory was 11,292 MiB, well above the 2 to 4 GB I estimated in the Model Selection Report. My guess is that the run keeps the full predicted and ground-truth fields in memory for saving, since the output file is 6.6 GB. I haven't profiled it. The job fits on a 16 GB V100 and may need a larger card than that if the data window grows.
- **Output size.** My first job script copied the 6.6 GB prediction file into `outputs/`. Now it stays on scratch, and I commit a small CSV of Z500 skill instead.
- **Git on ICE.** `git pull` refused to merge because my ICE copy of the job script had a local edit to the account line. `git checkout -- slurm/run_inference.sbatch` fixed it, since the repo version already holds `--account=coc`.

## Known risks

- **Stats and checkpoint pairing.** The `stats_v0` files have to match the checkpoint release. If ACC lands outside [-1, 1] or comes out very low at 24 h, check this first.
- **Version drift.** `inference.py` needs numpy below 2 and torch below 2.6, and `smoke_test.py` checks both.
- **Numerical differences.** Different GPUs and cuFFT versions will give slightly different numbers, so I compare curve shape with the paper and skip digit-level matching.
- **Job limits.** PACE-ICE caps users at roughly 50 GPU jobs and 32 GPU-hours, and this run uses well under an hour.
- **Old Docker base image.** The official image predates Hopper GPUs, which is why the environment pins newer wheels (see `environment/VERSION_NOTES.md`).
- **GPU memory.** Peak use is about 11.3 GiB with `--vis`, so request a GPU with at least 16 GB.

## Cleanup

- Remove the per-run output on scratch with `rm -rf ~/scratch/fcn-run-<jobid>` after copying the plot and logs into this repo.
- Remove the artifacts with `rm -rf ~/scratch/fcn-artifacts`. `scripts/download_artifacts.sh` fetches them again, and they stay out of the repo.
- Remove the environment with `conda env remove -p ~/scratch/.conda/envs/fourcastnet-inference`.
- Clear the caches with `pip cache purge; conda clean --all -y` to protect the home quota.

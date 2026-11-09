# FourCastNet reproducible inference on PACE-ICE

**Completed V100 run:** job **6205007**, finished October 12, 2026 at 1:58 AM EDT (05:58 UTC). FourCastNet ran a 234-hour rollout (40 saved states: the initial condition plus 39 six-hour steps) on a Tesla V100-PCIE-16GB on PACE-ICE, the cluster my course calls AI Makerspace, under account `coc`, partition `ice-gpu` and QOS `coc-ice`. All smoke tests and substantive checks passed. A second job (6205661) on the same node gave identical numbers. Raw evidence is in `logs/` and `outputs/`.

The job used 1 GPU, 8 CPU cores and 64 GB of requested RAM, of which 21.7 GB was used. It took **95 seconds** of wall time, 77 of them in inference. A rerun from a fresh clone of the PR branch (job 6885282, also a V100) took 243 seconds, 113 of them in inference, and gave identical results, so expect timings to vary with the node and the load. Peak GPU memory was **11,292 MiB**, sampled twice a second in `logs/gpu_usage_6205007.csv`.

This workflow loads the pretrained FourCastNet AFNO backbone (`backbone.ckpt` from the NVIDIA demo bundle) and rolls it forward 39 six-hour steps from the first state of the released 2018 out-of-sample ERA5 file. That gives 40 saved states, from lead 0 h (the initial condition) to lead 234 h. It scores 20 channels against ERA5 with RMSE and ACC, and Z500 (channel 14) is the headline variable. My model-selection issue also proposed T850, but `inference.py` only scores Z500, so T850 is not covered by this workflow. Z500 ACC is **0.9952 at 24 hours**, **0.9813 at 48 hours**, and it first falls below 0.6 at **174 hours**. Training, the precipitation model and the 26-channel HuggingFace release are out of scope.

Model-selection tracking issue: https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/24

## 1. Required resources

Use a GPU node on the `ice-gpu` partition with the course account. The job script asks for the resources below and leaves the GPU type open, so Slurm picks whichever card is free.

| Resource | Request | Measured |
|---|---|---|
| GPU | 1, with at least 16 GB of memory | Tesla V100-PCIE-16GB, peak 11,292 MiB |
| CPU cores | 8 | not measured |
| RAM | 64 GB | 21.7 to 22.0 GB |
| Wall time | 45 minutes | 95 s (job 6205007), 243 s (job 6885282) |
| Scratch storage | plan for about 20 GB | the artifact tar is 3,998,497,644 bytes and the prediction file is 6.6 GB |

I did not measure the final total for storage. The 20 GB figure is my estimate for the environment, artifacts and prediction file together. The home quota is about 30 GB and fills quickly, so put everything large on scratch. The workflow needs outbound HTTPS for the package install and the artifact download, and it needs no Globus account.

Software is Python 3.10, torch 2.2.2+cu121 and numpy 1.26.4, installed from `environment/environment.yml` into a conda environment called `fourcastnet-inference`. `inference/inference.py` forces two pins, numpy below 2 and torch below 2.6, and `environment/VERSION_NOTES.md` explains them. The cluster loads conda through `module load miniforge`.

## 2. Setup

Everything below runs on PACE-ICE. Connect to the GT VPN, then `ssh <gt-username>@login-ice.pace.gatech.edu`.

Find your Slurm account. You pass it on the command line when you submit, so no file needs editing. I used `coc`.

```bash
sacctmgr show associations user=$USER format=account,qos,partition
```

Clone the FourCastNet code at the commit I used, then clone this branch onto scratch and change into this folder. Every later command runs from this folder.

```bash
git clone https://github.com/NVlabs/FourCastNet.git ~/FourCastNet
git -C ~/FourCastNet checkout 93360c1720a9f97aabf970689f21c9fad8737788
git clone -b recipe-fourcastnet-ice-rollout https://github.com/srir1607/cybershuttle-ai4science-foundry.git ~/scratch/foundry
cd ~/scratch/foundry/registry/models/climate/fourcastnet/ice-v100-rollout
```

Create the environment on scratch. The `~/.condarc` file has to be valid YAML, so this command overwrites it. The `miniforge` module is loaded three times on purpose, because the repeats override the default anaconda module on PACE-ICE.

```bash
mkdir -p ~/scratch/.conda/envs
printf 'envs_dirs:\n  - ~/scratch/.conda/envs\n' > ~/.condarc
module purge; module load miniforge; module load miniforge; module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba env create -f environment/environment.yml -p ~/scratch/.conda/envs/fourcastnet-inference
conda activate fourcastnet-inference
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

The last line should print `2.2.2+cu121 1.26.4`. Next, download and verify the artifacts. The download is about 4 GB over plain HTTPS. The script resumes with `wget -c`, extracts the archive and checks every file against `docs/CHECKSUMS.sha256`, writing `logs/download_manifest.txt`. If the login node kills the transfer (exit code 137), run the same script inside `srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --time=01:00:00 --pty bash`.

```bash
./scripts/download_artifacts.sh ~/scratch/fcn-artifacts
```

Run the smoke test on the login node. It reports no CUDA device there, which is expected. Then check CUDA on a GPU node.

```bash
python scripts/smoke_test.py --fourcastnet-repo ~/FourCastNet --artifact-root ~/scratch/fcn-artifacts
srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --pty bash
conda activate fourcastnet-inference
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

## 3. Run inference

Submit from this folder, because the job script builds its paths from the submit directory.

```bash
sbatch --account=<acct> slurm/run_inference.sbatch
squeue -u $USER
```

The job patches `config/AFNO.yaml` and saves the diff to `logs/config_patch.diff`. It samples GPU memory into `logs/gpu_usage_<jobid>.csv`, runs `inference/inference.py --config=afno_backbone --run_num=0 --vis` for one initial condition, and then runs `scripts/validate.py`. The evaluation file holds 40 time steps and the default config expects 41, so `scripts/configure_paths.py` sets `prediction_length` from the file shape. The 40 saved states are the initial condition at lead 0 h plus 39 forecast steps, so the last one is at lead 234 h. The job stops at the start with an error if `~/FourCastNet` is not at the pinned commit `93360c1720a9f97aabf970689f21c9fad8737788`. It uses the shipped `backbone.ckpt` checkpoint as is, and no training or fine-tuning happens. The 6.6 GB prediction file goes to `~/scratch/fcn-run-<jobid>`, and the job exits with a non-zero code if validation fails.

## 4. Validate

`scripts/validate.py` runs at the end of the job. You can also run it alone with `python scripts/validate.py --run-dir ~/scratch/fcn-run-<jobid>`. The full record is in `docs/VALIDATION.md`.

| Check | Expected | Measured in job 6205007 |
|---|---|---|
| Output HDF5 holds `rmse` and `acc`, finite and non-empty | present | both have shape (1, 40, 20) |
| ACC within [-1, 1] on all 20 channels | by definition | min 0.0922, max 1.0000 |
| ACC at lead 0 | about 1 | 1.0000 |
| Z500 ACC at 24 h | at least 0.6 | 0.9952 |
| ACC falls and RMSE rises with lead time | steady trend | ACC 0.9952 to 0.2874, RMSE 79.61 to 1069.85 (24 h to 234 h) |
| Z500 ACC and RMSE match the reference run `outputs/z500_skill_6205007.csv` | ACC within 0.01 and RMSE within 10 at every lead time | job 6205661 differs by 0 on the same GPU, and an independent H100 run by 0.002741 (ACC) and 3.566 (RMSE) |

These are the Z500 values from job 6205007 that a rerun should reproduce. A correct run lands within the reference tolerances described below (ACC within 0.01 and RMSE within 10) at every lead time.

| Lead time | Z500 ACC | Z500 RMSE |
|---|---|---|
| 24 h | 0.9952 | 79.61 |
| 48 h | 0.9813 | 156.09 |
| 72 h | 0.9437 | 271.82 |
| 120 h | 0.8252 | 478.35 |

The 0.6 threshold is the usual convention for a useful forecast, and the FourCastNet paper does not define it. Passing it at 24 hours is easy for a working model, so the lead time where ACC first drops below 0.6 says more. Mine first falls below it at 174 hours, and so did the independent H100 rerun.

The reference comparison runs automatically at the end of the job through `--reference-csv outputs/z500_skill_6205007.csv` in `slurm/run_inference.sbatch`. The tolerances are 0.01 for ACC and 10 for RMSE at every lead time. I set them at roughly three times the largest differences seen when a classmate reran the workflow on an H100 (0.002741 for ACC and 3.566 for RMSE), because a different GPU or cuFFT version changes the last digits of a 40-state autoregressive rollout. Two GPU types is too little data to call these tolerances tight, so treat them as a starting point and widen them only after finding the cause of a failure.

I compared the curve with Figure 7(a) of Pathak et al. (2022), reading the paper's values off the figure by eye. The two agree through about 2 days, with ACC near 0.99 at 24 to 48 hours. After about 4 days my curve falls below the paper's average. At 192 hours my ACC is 0.443 against roughly 0.71, and at 234 hours it is 0.287 against roughly 0.61. The paper averages many forecasts and I ran one initial condition. The demo bundle holds a single 10-day window, so I could not test whether that explains the gap. I treat the shortfall beyond about 4 days as an open discrepancy.

To check repeatability, run the job twice and compare the CSVs with `python scripts/compare_runs.py outputs/z500_skill_<jobidA>.csv outputs/z500_skill_<jobidB>.csv`. Jobs 6205007 and 6205661 differed by 0 on every ACC and RMSE value. Both used the same node and GPU, so this says nothing about drift between GPU types or cuFFT versions. A classmate (prathampatel21) later reran the workflow on an H100 from a fresh environment and download. Z500 ACC was 0.995214 at 24 hours and 0.981347 at 48 hours against my saved 0.995209 and 0.981325. The largest difference over all 40 states was 0.002741 in ACC and 3.566 in RMSE, and both runs first fell below 0.6 at 174 hours.

## 5. Preserve execution evidence

After the job finishes, record the accounting data and keep the small outputs next to the logs.

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

The evidence for each job is `logs/slurm-<jobid>.out` and `.err`, `logs/inference_out_<jobid>.log`, `logs/gpu_usage_<jobid>.csv`, `outputs/z500_skill_<jobid>.csv` and `outputs/skill_vs_leadtime.png`. The 6.6 GB prediction file stays on scratch. `docs/PROVENANCE.md` traces every artifact and its checksum so someone else can rebuild the same run. Keep failed runs too, and look through logs for private paths before publishing.

## 6. Troubleshooting and cleanup

The problems I hit while building this, with their fixes, are in `docs/TROUBLESHOOTING.md`. The short version is below.

- `module load miniforge` printing a `ParserError` points at a broken `~/.condarc`. Rewrite it with `>` and a single valid `envs_dirs:` block.
- A `wget` that dies near 95% with exit code 137 was killed by the login node. Resume it with `wget -c`, or run the download inside an `srun` session.
- If home usage climbs toward the 30 GB quota, run `pip cache purge` and `conda clean --all -y`, and keep the environment and artifacts on scratch.
- Always request `--partition=ice-gpu`. The Blackwell cards in `ice-bw-gpu` need a newer torch build than the pinned 2.2.2, so I expect them to fail with "no kernel image".
- If ACC lands outside [-1, 1] or comes out very low at 24 hours, check that the `stats_v0` files match the checkpoint release.
- PACE-ICE caps users at roughly 50 GPU jobs and 32 GPU-hours, and this workflow uses well under an hour.

To clean up, remove the run output with `rm -rf ~/scratch/fcn-run-<jobid>` after copying the plot and logs, remove artifacts with `rm -rf ~/scratch/fcn-artifacts`, and remove the environment with `conda env remove -p ~/scratch/.conda/envs/fourcastnet-inference`. The download script fetches the artifacts again when you need them.

## Sources and licenses

- Paper: Pathak et al. (2022), https://arxiv.org/abs/2202.11214
- Code: https://github.com/NVlabs/FourCastNet at commit `93360c1720a9f97aabf970689f21c9fad8737788` (BSD 3-Clause)
- Data and weights: https://portal.nersc.gov/project/m4134/ccai_demo.tar, published by the FourCastNet authors. File sizes and SHA-256 checksums are in `docs/CHECKSUMS.sha256`.
- The underlying data is ERA5 reanalysis from ECMWF and Copernicus. The file is the authors' preprocessed out-of-sample subset, and I downloaded no raw ERA5. Check ECMWF and Copernicus terms before redistributing it.

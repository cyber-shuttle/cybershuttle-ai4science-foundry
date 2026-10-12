# ClimaX reproducible inference on PACE ICE

This repository branch documents and implements a Georgia Tech AI Makerspace / PACE ICE workflow for fine-tuning ClimaX's public pretrained model on ERA5, running global forecast inference, and validating the predictions. **Start with this guide.** For full dataset instructions, troubleshooting, and evidence details, see the [complete workflow guide](workflows/climax-global-forecast/README.md).

## Reproduction target

- **Task:** global ERA5 forecast at 5.625° resolution, predicting 72 hours ahead.
- **Model source:** [Microsoft ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c).
- **Initialization checkpoint:** [`tungnd/climax/5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt), Hugging Face revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`.
- **Data:** WeatherBench ERA5, retrieved externally and converted using the pinned ClimaX preprocessing script.
- **Important:** `5.625deg.ckpt` is a pretraining checkpoint, not a fine-tuned forecaster. The reproducible sequence fine-tunes it with ClimaX's global-forecast trainer before inference.

## Start-to-finish sequence

### 1. Connect and clone the assignment branch

Log in to PACE ICE using the current [PACE ICE access guide](https://docs.pace.gatech.edu/ice/). Clone the published assignment branch into scratch:

```bash
mkdir -p "$HOME/scratch"
git clone --branch climax-inference \
  https://github.com/purvikathalkar/climax-inference.git \
  "$HOME/scratch/climax-inference"
```

If the branch is not available yet, stop and use the published assignment branch when it is pushed; do not silently substitute `main`.

### 2. Initialize Miniforge and put environments on scratch

On a fresh shell, load Miniforge before running any Conda command. Set the scratch environment directory in your Conda user configuration once:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda --version
mkdir -p "$HOME/scratch/.conda/envs"
conda config --add envs_dirs "$HOME/scratch/.conda/envs"
conda config --show envs_dirs
```

This allows `conda activate climax-inference` by name while avoiding the small home quota. If Miniforge is not found, check `module avail miniforge` / `module spider miniforge` and current PACE documentation rather than installing a second Conda distribution.

### 3. Create and test the pinned environment

Allocate a GPU before testing CUDA; a login node does not expose a GPU:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" --partition=ice-gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

Inside the allocation:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
bash scripts/setup_env.sh
conda activate climax-inference
```

The environment pins Python 3.8.16, PyTorch 1.12.1, torchvision 0.13.1, CUDA runtime 11.3.1, Lightning 1.8.0, and the other dependencies in [`environment.yml`](workflows/climax-global-forecast/environment.yml). Setup verifies training imports and performs a real CUDA tensor operation. Continue only if that check succeeds. The older stack may not support newer GPU architectures; see the detailed guide's [environment and GPU checks](workflows/climax-global-forecast/README.md#4-initialize-conda-and-create-the-environment).

The helper uses Mamba when it is available (otherwise Conda is used and may take longer). The environment pins `pip=24.0` and `mkl=2024.0`; the helper also reapplies these pins after environment creation/update because `lightning-lite==1.8.0` is incompatible with newer pip releases in this stack.

### 4. Retrieve source code and pretrained weights

```bash
mkdir -p "$HOME/scratch/climax-reproduction"
cd "$HOME/scratch/climax-reproduction"
git clone https://github.com/microsoft/ClimaX.git
git -C ClimaX checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
mkdir -p checkpoints
curl --fail --location --retry 3 --retry-all-errors \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  --output checkpoints/5.625deg.ckpt
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  checkpoints/5.625deg.ckpt' \
  | sha256sum --check -
```

Expected checkpoint size: 432,419,871 bytes. `sha256sum` must report `OK`; otherwise stop and download again. The checkpoint and model source are external to this repository.

### 5. Retrieve and preprocess ERA5

Download WeatherBench 5.625° ERA5 from the [TUM archive linked in ClimaX documentation](https://dataserv.ub.tum.de/index.php/s/m1524895), following its current access instructions. Stage/extract the original files under `$HOME/scratch/climax-reproduction/data/5.625deg`. If the archive cannot be downloaded directly from PACE, download it on a machine with access and copy it to scratch. Check `constants.nc`, all required variable directories, and the year coverage before preprocessing.

The raw tree needs `constants.nc` and NetCDF directories for `2m_temperature`, both 10-metre wind components, and pressure-level geopotential, wind, temperature, relative humidity, and specific humidity. Request a CPU allocation using a partition available to your account (check current PACE guidance):

```bash
srun --account="<YOUR_PACE_ACCOUNT>" --partition="<YOUR_CPU_PARTITION>" \
  --cpus-per-task=16 --mem=128G --time=12:00:00 --pty bash
```

In the allocated CPU shell:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate climax-inference
python "$HOME/scratch/climax-reproduction/ClimaX/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$HOME/scratch/climax-reproduction/data/5.625deg" \
  --save_dir "$HOME/scratch/climax-reproduction/data/5.625deg_npz" \
  --start_train_year 1979 --start_val_year 2016 \
  --start_test_year 2017 --end_year 2019 --num_shards 8 \
  --variables 2m_temperature \
  --variables 10m_u_component_of_wind \
  --variables 10m_v_component_of_wind \
  --variables geopotential \
  --variables u_component_of_wind \
  --variables v_component_of_wind \
  --variables temperature \
  --variables relative_humidity \
  --variables specific_humidity
```

The year ranges are train 1979–2015, validation 2016, and test 2017–2018 (`end_year` is exclusive). The detailed guide lists the expected prepared files and how to confirm them before allocating a GPU job. Raw ERA5 is not stored in this repository.

### 6. Fine-tune for forecasting

Return to a login shell after preprocessing. A fresh shell needs Miniforge initialized again, and the PACE account is the only user-specific value below. Create the log directory before submission because SLURM opens output files before the script starts:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/finetune.sbatch
squeue --me
```

The job uses the documented scratch layout by default and requests one `ice-gpu` GPU, 8 CPUs, 64 GiB RAM, and 24 hours. Adjust only for resources permitted by your allocation and document deviations. Checkpoints and training logs go under `$HOME/scratch/climax-reproduction/runs/global-forecast`; retain the submitted job ID and SLURM output/error logs.

When training completes, identify the **validation-selected fine-tuned checkpoint** from the checkpoint restore message and checkpoint directory. Do not pass the original pretraining checkpoint as the inference model.

### 7. Run held-out inference

Pass the exact checkpoint selected by validation as the job argument (replace `epoch_XXX.ckpt` with the actual filename):

```bash
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/infer.sbatch \
  "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints/epoch_XXX.ckpt"
squeue --me
```

The job evaluates 32 daily-spaced samples from a test shard at a 72-hour lead. Results default to `$HOME/scratch/climax-reproduction/runs/inference-<job-id>`.

### 8. Inspect files and validate metrics

After the inference job completes:

```bash
JOB_ID="<inference-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat "$HOME/scratch/climax-reproduction/runs/inference-$JOB_ID/metrics.json"
ls -lh "$HOME/scratch/climax-reproduction/runs/inference-$JOB_ID"
```

Expected files:

- `climax-forecast.npz`: predictions, ERA5 targets, persistence/climatology baselines, coordinates, and sample metadata.
- `metrics.json`: shape/finite smoke check, per-variable latitude-weighted RMSE, relative improvement against persistence, ranges, checkpoint hashes, and run metadata.
- `logs/climax-inference-<JOB_ID>.out` / `.err`: PACE execution logs.

The smoke check must pass with prediction shape `[32, 5, 32, 64]` and finite predictions/targets. In the metric table, positive `climax_vs_persistence_relative_improvement` means lower RMSE than persistence; negative means persistence did better. Compare every output variable with both baselines and report all results. Check the full workflow guide for broad physical-range checks and the proper interpretation of the test climatology.

There is **no official absolute RMSE cutoff for this exact run** in the pinned upstream instructions; data version, trained checkpoint, selected evaluation windows, and training scope all affect the score. Do not invent a pass threshold. Use the upstream held-out test metrics and relative baseline comparison as evidence, and claim an advantage only where the measured values support it.

### 9. Preserve evidence and clean up

Complete the [execution record](workflows/climax-global-forecast/README.md#10-execution-evidence-and-run-record) in the workflow guide with the date, job IDs/states, source commit, checkpoint hashes, environment export, data/preprocessing details, actual GPU/resources/runtime, observed metrics, and failures/deviations. Include useful logs and a small reviewed result artifact if repository/data policy permits. Do not commit ERA5 or checkpoint files.

Only after the evidence is saved and reviewed, optionally remove specifically named run outputs/checkpoints or the Conda prefix. See the detailed [cleanup instructions](workflows/climax-global-forecast/README.md#12-cleanup); do not recursively delete `$HOME/scratch`.

## Hardware and outputs at a glance

| Stage | Starting request | Main output |
|---|---|---|
| Environment setup | In allocated GPU shell | Conda/Python version records |
| ERA5 preprocessing | CPU allocation, suggested 16 CPUs / 128 GiB RAM; verify partition/quota | Shards, normalization, climatologies, coordinates |
| Forecast fine-tuning | 1 GPU, 8 CPUs, 64 GiB RAM, 24h | Lightning checkpoints and training/validation/test logs |
| Inference | 1 GPU, 4 CPUs, 32 GiB RAM, 1h | `.npz` predictions and `metrics.json` |

Requests are starting points, not verified measurements. Always use the resource limits and partitions assigned by PACE.

## Supporting documentation

- [Complete ClimaX workflow and troubleshooting guide](workflows/climax-global-forecast/README.md)
- [PACE ICE docs: login, modules, storage, SLURM](https://docs.pace.gatech.edu/ice/)
- [ClimaX usage guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [ClimaX global forecast config](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml)
- [WeatherBench ERA5 archive](https://dataserv.ub.tum.de/index.php/s/m1524895)

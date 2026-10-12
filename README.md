# ClimaX reproducible inference on PACE ICE

This repository branch documents and implements a Georgia Tech AI Makerspace / PACE ICE workflow for fine-tuning ClimaX's public pretrained model on ERA5, running global forecast inference, and validating the predictions. **Start with this guide.** For full dataset instructions, troubleshooting, and evidence details, see the [complete workflow guide](workflows/climax-global-forecast/README.md).

## Reproduction target

- **Task:** global ERA5 forecast at 5.625° resolution, predicting 72 hours ahead.
- **Model source:** [Microsoft ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c).
- **Initialization checkpoint:** [`tungnd/climax/5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt), Hugging Face revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`.
- **Data:** WeatherBench ERA5, retrieved externally and converted using the pinned ClimaX preprocessing script.
- **Important:** `5.625deg.ckpt` is a pretraining checkpoint, not a fine-tuned forecaster. The reproducible sequence fine-tunes it with ClimaX's global-forecast trainer before inference.

No PACE run has been executed from this checkout yet. GPU model, execution time, memory use, and measured forecast errors must come from the student's real run and are not fabricated in this repository.

## Start-to-finish sequence

### 1. Connect and clone the assignment branch

Log in to PACE ICE and replace the placeholders with your assigned values:

```bash
export PACE_ACCOUNT="<your-PACE-account>"
mkdir -p "$HOME/scratch"
export WORKFLOW_REPO="$HOME/scratch/climax-inference"
git clone --branch climax-inference \
  https://github.com/purvikathalkar/climax-inference.git "$WORKFLOW_REPO"
export WORKFLOW_DIR="$WORKFLOW_REPO/workflows/climax-global-forecast"
cd "$WORKFLOW_DIR"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export ENV_PREFIX="$HOME/scratch/.conda/envs/climax-inference"
```

The branch must first be pushed to GitHub for a new student to clone it this way.

### 2. Get a GPU shell and initialize Conda

On PACE, `conda` is provided by the Miniforge module; it may not be in a fresh shell's `PATH`. **Load the module and source its shell initialization before using Conda:**

```bash
srun --account="$PACE_ACCOUNT" --partition=ice-gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
command -v conda
conda --version
```

If the module is unavailable, use `module avail miniforge` / `module spider miniforge` and follow current PACE module instructions; do not assume Conda is preinstalled or guess an installation path. For exact setup and storage/cache handling, see [environment setup](workflows/climax-global-forecast/README.md#4-load-conda-correctly-and-create-the-environment).

### 3. Create the pinned environment

From the workflow directory, still in the GPU allocation:

```bash
cd "$WORKFLOW_DIR"
bash scripts/setup_env.sh
```

The environment pins Python 3.8.16, PyTorch 1.12.1, torchvision 0.13.1, CUDA runtime 11.3.1, PyTorch Lightning 1.8.0, `jsonargparse` 4.19.0, Rich 13.3.1, TensorBoard 2.11.2, timm 0.6.12, and preprocessing dependencies in [`environment.yml`](workflows/climax-global-forecast/environment.yml). The helper redirects Conda/pip/temp/model caches to scratch, import-checks the training runtime, and writes environment exports under `execution/`. CUDA should be available inside the allocated GPU job. This upstream-era stack is not guaranteed to support Hopper/H100-or-newer; see the hardware caveat in the detailed guide before proceeding on newer GPUs.

### 4. Retrieve model source and initialization weights

In the workflow's parent scratch project directory:

```bash
cd "$PROJECT_DIR"
git clone https://github.com/microsoft/ClimaX.git
git -C ClimaX checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"

mkdir -p "$PROJECT_DIR/checkpoints"
export PRETRAINED_CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
curl --fail --location --retry 3 --retry-all-errors \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  --output "$PRETRAINED_CHECKPOINT"
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$PRETRAINED_CHECKPOINT" \
  | sha256sum --check -
```

Expected checkpoint size: 432,419,871 bytes. `sha256sum` must report `OK`; otherwise stop and download again. The checkpoint and model source are external to this repository.

### 5. Retrieve and preprocess ERA5

Download the WeatherBench 5.625° ERA5 archive from the [TUM archive linked in ClimaX documentation](https://dataserv.ub.tum.de/index.php/s/m1524895) and stage/extract it to:

```bash
export ERA5_RAW="$PROJECT_DIR/data/5.625deg"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"
```

The raw tree needs `constants.nc` and NetCDF directories for `2m_temperature`, both 10-metre wind components, and pressure-level geopotential, wind, temperature, relative humidity, and specific humidity. In a Conda-initialized CPU compute allocation with adequate storage/RAM:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$ERA5_RAW" --save_dir "$ERA5_NPZ" \
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

From a login shell where the workflow and scratch paths are set, create the log folder and submit the upstream forecast trainer:

```bash
mkdir -p "$WORKFLOW_DIR/logs"
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
export PRETRAINED_CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"
export WORKFLOW_DIR PROJECT_DIR ENV_PREFIX CLIMAX_SOURCE PRETRAINED_CHECKPOINT ERA5_NPZ
export TRAIN_DIR="$PROJECT_DIR/runs/global-forecast"
cd "$WORKFLOW_DIR"
sbatch --export=ALL --account="$PACE_ACCOUNT" slurm/finetune.sbatch
squeue --me
```

The job requests one `ice-gpu` GPU, 8 CPUs, 64 GiB RAM, and 24 hours; change only to resources permitted by your PACE allocation and record any changes. It runs up to 50 epochs with the upstream global-forecast config, initializes from the verified pretraining checkpoint, and writes validation-selected checkpoints and trainer logs under `$TRAIN_DIR`. These resource/time requests have not been measured in this checkout. Save the SLURM job ID and `.out`/`.err` logs.

When training completes, identify the **validation-selected fine-tuned checkpoint** from the checkpoint restore message and checkpoint directory. Do not pass the original pretraining checkpoint as the inference model.

### 7. Run held-out inference

Set the fine-tuned checkpoint path to the exact checkpoint selected by validation:

```bash
export FINETUNED_CHECKPOINT="/absolute/path/to/the/validation-selected/epoch_checkpoint.ckpt"
export OUTPUT_DIR="$PROJECT_DIR/runs/inference-test-2017"
cd "$WORKFLOW_DIR"
sbatch --export=ALL --account="$PACE_ACCOUNT" slurm/infer.sbatch
squeue --me
```

The job evaluates 32 daily-spaced samples from a test shard at a 72-hour lead. The same workflow guide includes an interactive command if you have an allocated GPU shell and want to run without `sbatch`.

### 8. Inspect files and validate metrics

After the inference job completes:

```bash
JOB_ID="<inference-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat "$OUTPUT_DIR/metrics.json"
ls -lh "$OUTPUT_DIR"
```

Expected files:

- `climax-forecast.npz`: predictions, ERA5 targets, persistence/climatology baselines, coordinates, and sample metadata.
- `metrics.json`: shape/finite smoke check, per-variable latitude-weighted RMSE, relative improvement against persistence, ranges, checkpoint hashes, and run metadata.
- `logs/climax-inference-<JOB_ID>.out` / `.err`: PACE execution logs.

The smoke check must pass with prediction shape `[32, 5, 32, 64]` and finite predictions/targets. In the metric table, positive `climax_vs_persistence_relative_improvement` means lower RMSE than persistence; negative means persistence did better. Compare every output variable with both baselines and report all results. Check the full workflow guide for broad physical-range checks and the proper interpretation of the test climatology.

There is **no official absolute RMSE cutoff for this exact run** in the pinned upstream instructions; data version, trained checkpoint, selected evaluation windows, and training scope all affect the score. Do not invent a pass threshold. Use the upstream held-out test metrics and relative baseline comparison as evidence, and claim an advantage only where the measured values support it.

### 9. Preserve evidence and clean up

Complete [`execution/README.md`](workflows/climax-global-forecast/execution/README.md) with the date, job IDs/states, source commit, checkpoint hashes, environment export, data/preprocessing details, actual GPU/resources/runtime, observed metrics, and failures/deviations. Include useful logs and a small reviewed result artifact if repository/data policy permits. Do not commit ERA5 or checkpoint files.

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

# ClimaX inference on PACE ICE

This repository's primary workflow runs inference directly with Microsoft's public ClimaX pretrained weights. It extracts spatial and pooled representations from ERA5 geopotential at 500 hPa; it does **not** fine-tune ClimaX and does **not** claim to forecast weather. This narrower inference task avoids training and needs only one pressure-level input group plus static constants, rather than the full ERA5 input set.

For detailed commands, troubleshooting, cleanup, and the optional forecast experiment, see the [workflow guide](workflows/climax-global-forecast/README.md).

## Reproduction target and provenance

- **Task:** extract pretrained ClimaX encoder embeddings from `geopotential_500` on the WeatherBench 5.625° grid.
- **Model code:** [Microsoft ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c).
- **Weights:** [`tungnd/climax/5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt), pinned Hugging Face revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`, size 432,419,871 bytes.
- **Data:** WeatherBench ERA5 `geopotential` pressure-level group plus `constants.nc`; only the 500 hPa geopotential channel is passed to the encoder. The pilot split below uses a short year range.
- **Outputs:** `climax-embeddings.npz` contains spatial tokens `[N, 512, 1024]` and mean-pooled vectors `[N, 1024]`; `metrics.json` records shapes, finite/nonconstant checks, deterministic repeat inference, hashes, and runtime.

ClimaX's upstream global-forecast example uses this checkpoint to initialize a different downstream forecasting task and then fine-tunes it. This repository's primary embedding task instead calls the pretrained encoder directly and strictly loads its pretraining checkpoint. Do not interpret the generated vectors as 72-hour predictions.

## Step-by-step reproduction

### 1. Connect, clone, and initialize Miniforge

Log into PACE ICE using the [current access documentation](https://docs.pace.gatech.edu/ice/), then clone the assignment branch to scratch:

```bash
mkdir -p "$HOME/scratch"
git clone --branch climax-inference \
  https://github.com/purvikathalkar/climax-inference.git \
  "$HOME/scratch/climax-inference"
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda --version
mkdir -p "$HOME/scratch/.conda/envs"
conda config --add envs_dirs "$HOME/scratch/.conda/envs"
```

If `conda` is missing, initialize Miniforge as above in the current shell. Do not install another Conda distribution over the cluster-provided module.

### 2. Create the environment and verify GPU support

Start an allocation using your assigned account and a currently available GPU partition:

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
python --version
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

Expected versions: Python `3.8.16`, PyTorch `1.12.1`, CUDA runtime `11.3.1`, and `CUDA available: True` inside the GPU allocation. The helper uses Mamba if available, pins pip 24.0 and MKL 2024.0, and records Conda/pip inventories under `workflows/climax-global-forecast/execution/`. Full pins are in [`environment.yml`](workflows/climax-global-forecast/environment.yml).

### 3. Clone the pinned model source and retrieve pretrained weights

```bash
mkdir -p "$HOME/scratch/climax-reproduction/checkpoints"
git clone https://github.com/microsoft/ClimaX.git \
  "$HOME/scratch/climax-reproduction/ClimaX"
git -C "$HOME/scratch/climax-reproduction/ClimaX" \
  checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
curl --fail --location --retry 3 --retry-all-errors \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  --output "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt"
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt" \
  | sha256sum --check -
```

Continue only if the checksum reports `OK`.

### 4. Download and preprocess the reduced ERA5 input

Download the constants and geopotential folders directly from the TUM WeatherBench archive on ICE. The geopotential pressure-level group is required because the pinned ClimaX preprocessor expects the parent group and extracts the pressure levels it needs. Although WeatherBench also provides a separate geopotential_500 ZIP, that archive does not use the expected input layout.

The following commands download the two folders as ZIP archives, extract their contents, and stage the files under $HOME/scratch/climax-reproduction/data/5.625deg/.

##### 1. Create the download and data directories

```bash
DOWNLOAD_DIR="$HOME/scratch/climax-reproduction/downloads"
DATA_DIR="$HOME/scratch/climax-reproduction/data/5.625deg"
mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR/geopotential"
cd "$DOWNLOAD_DIR"
```

##### 2. Download the constants archive

```bash
curl -fL --retry 3 \
  -o constants.zip \
  'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fconstants'
```

##### 3. Download the geopotential archive

```bash
curl -fL --retry 3 \
  -o geopotential.zip \
  'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fgeopotential'
```

These commands save the archives to scratch storage instead of your home directory. The geopotential download may be large, so check its size before starting if storage is limited:

```bash
du -h constants.zip geopotential.zip
df -h "$HOME/scratch"
```

##### 4. Verify the archives before extracting

```bash
file constants.zip geopotential.zip
unzip -t constants.zip
unzip -t geopotential.zip
```

Both files should be valid ZIP archives, and both integrity tests should finish successfully.

##### 5. Extract the archives

```bash
mkdir -p "$DOWNLOAD_DIR/constants" "$DOWNLOAD_DIR/geopotential"
unzip -n constants.zip -d "$DOWNLOAD_DIR/constants"
unzip -n geopotential.zip -d "$DOWNLOAD_DIR/geopotential"
```

##### 6. Stage the NetCDF files

Locate the extracted constants file:

```bash
find "$DOWNLOAD_DIR/constants" -type f -name 'constants.nc'
```

Copy it to the expected location:

```bash
CONSTANTS_FILE="$(find "$DOWNLOAD_DIR/constants" -type f -name 'constants.nc' -print -quit)"
if [ -z "$CONSTANTS_FILE" ]; then
    echo "ERROR: constants.nc was not found in the extracted archive."
    exit 1
fi
cp -n "$CONSTANTS_FILE" "$DATA_DIR/constants.nc"
```

Copy the extracted geopotential NetCDF files into the expected geopotential directory while preserving their filenames:

```bash
find "$DOWNLOAD_DIR/geopotential" -type f -name '*.nc' \
  -exec cp -n -t "$DATA_DIR/geopotential" {} +
```

##### 7. Verify the final directory structure

```bash
find "$DATA_DIR" -maxdepth 2 -type f
du -sh "$DATA_DIR"
```

The expected structure is:

```text
$HOME/scratch/climax-reproduction/data/5.625deg/
├── constants.nc
└── geopotential/
    ├── <original NetCDF filename>.nc
    └── ...
```

The exact filenames depend on the archive. Preserve them rather than renaming the files manually.

##### 8. Preprocess the reduced input

Before running the pinned ClimaX preprocessing script, verify that the downloaded files contain the required years and pressure levels, and confirm that the script supports processing this reduced variable set. The intended pilot period is 2013–2017, but the preprocessing configuration must match the available data and the selected inference task.

Note: Downloading only constants and geopotential is appropriate only if the selected inference workflow can operate with that reduced input. Confirm the model’s required input variables before running preprocessing.

### 4. Download and preprocess the reduced ERA5 input

From the [TUM WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895), retrieve the `5.625deg/geopotential` pressure-level group and `constants.nc`. The WeatherBench README also mentions a separate `geopotential_500` ZIP, but the pinned ClimaX preprocessor expects the parent `geopotential` group and extracts its pressure levels. The selected input group is much smaller than downloading all ERA5 variables; the exact transfer size depends on the archive's download options.

Stage the files in this layout (preserve the archive's NetCDF filenames):

```text
$HOME/scratch/climax-reproduction/data/5.625deg/
  constants.nc
  geopotential/*.nc
```

Request a CPU allocation available to your account (suggested starting request, adjust to PACE policy):

```bash
srun --account="<YOUR_PACE_ACCOUNT>" --partition="<YOUR_CPU_PARTITION>" \
  --cpus-per-task=16 --mem=128G --time=12:00:00 --pty bash
```

Inside the allocation, initialize Miniforge and preprocess only geopotential:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate climax-inference
python "$HOME/scratch/climax-reproduction/ClimaX/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$HOME/scratch/climax-reproduction/data/5.625deg" \
  --save_dir "$HOME/scratch/climax-reproduction/data/geopotential_npz" \
  --start_train_year 2013 --start_val_year 2016 \
  --start_test_year 2017 --end_year 2018 --num_shards 8 \
  --variables geopotential
```

`end_year` is exclusive: this small demonstration uses train 2013–2015, validation 2016, and test 2017. It is a reduced-data representation run, not a full-data weather-forecast benchmark. The prepared data needs `train/`, `val/`, and `test/` `.npz` shards, `normalize_mean.npz`, and `normalize_std.npz`. Confirm that the test shard contains `geopotential_500` before submitting inference.

### 5. Run frozen pretrained embedding inference

Return to the login shell, initialize Miniforge there, and submit:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

The job requests one GPU, 4 CPUs, 32 GiB RAM, and one hour. It strictly loads the pinned checkpoint into the upstream pretraining module, encodes 32 daily-spaced samples from the first sorted test shard, and writes outputs to `$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>/`.

To run interactively instead, use the same arguments as `slurm/embed.sbatch` with `python scripts/embed.py` from the workflow directory. See the full [workflow guide](workflows/climax-global-forecast/README.md) for exact invocation, expected data checks, and troubleshooting.

### 6. Verify and preserve evidence

```bash
JOB_ID="<embedding-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID/metrics.json"
ls -lh "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID"
```

Expected output:

- `climax-embeddings.npz`: spatial token features, pooled vectors, variable name, and source sample indices.
- `metrics.json`: exact dimensions, all-finite/nonconstant checks, deterministic repeat check, model/checkpoint/config provenance, GPU, and runtime.
- `logs/climax-embed-<job-id>.out` and `.err`: SLURM logs.

The smoke check expects spatial embeddings `[32, 512, 1024]`, pooled vectors `[32, 1024]`, finite and nonconstant values, and identical repeated inference for the first input. These checks verify a reproducible non-degenerate encoder execution; they do not establish forecast skill or a published benchmark score. Preserve the logs, checkpoint hash, data period, output hash, actual GPU, and SLURM resource/runtime report.

## Scope and limitations

- This no-fine-tuning workflow extracts representations; it does not produce future weather fields.
- The input is intentionally reduced to `geopotential_500` and a short ERA5 period to reduce data and compute requirements.
- Representation non-degeneracy and deterministic repeat checks are technical validation, not a downstream scientific skill metric.
- The original 72-hour forecast workflow remains available in [`slurm/finetune.sbatch`](workflows/climax-global-forecast/slurm/finetune.sbatch) and [`slurm/infer.sbatch`](workflows/climax-global-forecast/slurm/infer.sbatch); it requires the full forecast input variables and task fine-tuning, and is separate from this primary workflow.
- Do not commit large data, checkpoints, or private PACE logs unless repository policy permits. Keep run evidence in an approved location.

## Supporting documentation

- [Detailed workflow, troubleshooting, and cleanup](workflows/climax-global-forecast/README.md)
- [PACE ICE documentation](https://docs.pace.gatech.edu/ice/)
- [Pinned ClimaX pretraining configuration](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/pretrain_climax.yaml)
- [Pinned ClimaX usage guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) and [WeatherBench README](https://github.com/pangeo-data/WeatherBench)

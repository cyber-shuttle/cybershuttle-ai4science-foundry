# ClimaX inference on PACE ICE

This repository's primary workflow runs inference directly with Microsoft's public ClimaX pretrained weights. It extracts spatial and pooled representations from ERA5 geopotential at 500 hPa; it does **not** fine-tune ClimaX and does **not** claim to forecast weather. This narrower inference task avoids training and needs only one pressure-level input group plus static constants, rather than the full ERA5 input set.

For detailed commands, troubleshooting, cleanup, and the optional forecast experiment, see the [workflow guide](workflows/climax-global-forecast/README.md).

## Reproduction target and provenance

- **Task:** extract pretrained ClimaX encoder embeddings from `geopotential_500` on the WeatherBench 5.625° grid.
- **Model code:** [Microsoft ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c).
- **Weights:** [`tungnd/climax/5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt), pinned Hugging Face revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`, size 432,419,871 bytes.
- **Data:** WeatherBench ERA5 `geopotential` pressure-level group plus `constants.nc`; only the 500 hPa geopotential channel is passed to the encoder. The pilot split below uses a short year range.
- **Outputs:** `climax-embeddings.npz` contains spatial tokens `[N, 512, 1024]` and mean-pooled vectors `[N, 1024]`; `metrics.json` records shapes, finite/nonconstant checks, deterministic repeat inference, and artifact hashes.

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
  --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
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

Download the `constants` and `geopotential` folders from the [TUM WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) directly to ICE scratch storage.

The `geopotential` archive contains pressure-level data. The pinned ClimaX preprocessing script expects the parent `geopotential` group, rather than the separate `geopotential_500` archive. The TUM downloads may contain nested ZIP files, so extract each archive before staging its NetCDF files.

#### 1. Create the download and data directories

```bash
export DOWNLOAD_DIR="$HOME/scratch/climax-reproduction/downloads"
export DATA_DIR="$HOME/scratch/climax-reproduction/data/5.625deg"

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR/geopotential"
cd "$DOWNLOAD_DIR"
```

#### 2. Download the archives

Download the constants archive:

```bash
curl -fL --retry 3 \
  -o constants.zip \
  'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fconstants'
```

Download the geopotential archive:

```bash
curl -fL --retry 3 \
  -o geopotential.zip \
  'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fgeopotential'
```

These commands save the archives to scratch storage. Before starting a large transfer, check available storage:

```bash
du -h constants.zip geopotential.zip
df -h "$HOME/scratch"
```

If a download is interrupted, rerun the command to restart it. The commands above do not resume partial downloads.

#### 3. Verify the downloaded archives

```bash
file constants.zip geopotential.zip
unzip -t constants.zip
unzip -t geopotential.zip
```

Both archives should pass their ZIP integrity tests before extraction.

#### 4. Extract the outer archives

```bash
mkdir -p "$DOWNLOAD_DIR/constants" "$DOWNLOAD_DIR/geopotential"

unzip -n constants.zip -d "$DOWNLOAD_DIR"
unzip -n geopotential.zip -d "$DOWNLOAD_DIR"
```

The constants archive contains `constants/constants_5.625deg.nc`. The geopotential archive contains `geopotential/geopotential_5.625deg.zip`, which must be extracted separately.

#### 5. Extract the nested geopotential archive

```bash
unzip -n \
  "$DOWNLOAD_DIR/geopotential/geopotential_5.625deg.zip" \
  -d "$DOWNLOAD_DIR/geopotential"
```

Inspect the extracted files:

```bash
find "$DOWNLOAD_DIR/constants" -maxdepth 3 -type f
find "$DOWNLOAD_DIR/geopotential" -maxdepth 3 -type f
```

If the nested archive contains additional ZIP files rather than individual NetCDF files, inspect those archives before proceeding.

#### 6. Stage the files in the expected directory structure

Copy the constants file to the canonical filename expected by the workflow:

```bash
CONSTANTS_FILE="$DOWNLOAD_DIR/constants/constants_5.625deg.nc"

if [ ! -f "$CONSTANTS_FILE" ]; then
    echo "ERROR: Expected constants file not found: $CONSTANTS_FILE"
    exit 1
fi

cp -n "$CONSTANTS_FILE" "$DATA_DIR/constants.nc"
```

Copy the extracted NetCDF files into the geopotential directory, preserving their original filenames:

```bash
find "$DOWNLOAD_DIR/geopotential" -type f -name '*.nc' \
  -exec cp -n -t "$DATA_DIR/geopotential" {} +
```

The `-n` flag prevents overwriting existing files.

#### 7. Verify the staged dataset

```bash
find "$DATA_DIR" -maxdepth 2 -type f
du -sh "$DATA_DIR"
```

The expected layout is:

```text
$HOME/scratch/climax-reproduction/data/5.625deg/
├── constants.nc
└── geopotential/
    ├── <original NetCDF filename>.nc
    └── ...
```

Verify the years and pressure levels represented in the NetCDF files before preprocessing. The archive's internal filenames and contents determine which years are available.

#### 8. Preprocess the reduced input, if required

**Important:** The project now targets pretrained ClimaX representation extraction, not the original 72-hour forecasting benchmark. Do not assume the forecasting preprocessing script can produce a valid representation-extraction dataset from geopotential alone. First verify the selected model's expected input variables and the preprocessing script's requirements.

If the pinned preprocessing script supports the reduced dataset, request a CPU allocation consistent with PACE policy. For example:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" \
  --partition="<YOUR_CPU_PARTITION>" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --pty bash
```

Inside the allocation, initialize the environment:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate climax-inference

export CLIMAX_SOURCE="$HOME/scratch/climax-reproduction/ClimaX"
export PYTHONPATH="$CLIMAX_SOURCE/src:$PYTHONPATH"
```

Check the supported arguments before running preprocessing:

```bash
python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" --help
```

Only if the script supports the selected variables and input layout, run:

```bash
python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$DATA_DIR" \
  --save_dir "$HOME/scratch/climax-reproduction/data/geopotential_npz_2016_test" \
  --start_train_year 2013 \
  --start_val_year 2015 \
  --start_test_year 2016 \
  --end_year 2017 \
  --num_shards 8 \
  --variables geopotential
```

Here, `end_year` is exclusive: training uses 2013–2014, validation uses 2015, and the test split uses 2016. This matches the successful `2016_0.npz` test shard. A 2017 NetCDF file encountered an HDF read error during preparation; do not represent the run as covering 2017 or as a full-data benchmark.

Before using the resulting dataset, verify that it contains the expected shards, normalization statistics, coordinate arrays, and required input variables for the selected representation-extraction implementation. Do not submit the original forecasting inference script unless its checkpoint and input requirements have also been satisfied.

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
sacct -j "$JOB_ID" --format=JobID,JobName,State,AllocTRES,MaxRSS,ExitCode
cat "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID/metrics.json"
ls -lh "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID"
```

Expected output:

- `climax-embeddings.npz`: spatial token features, pooled vectors, variable name, and source sample indices.
- `metrics.json`: exact dimensions, all-finite/nonconstant checks, deterministic repeat check, model/checkpoint/config provenance, and GPU.
- `logs/climax-embed-<job-id>.out` and `.err`: SLURM logs.

The smoke check expects spatial embeddings `[32, 512, 1024]`, pooled vectors `[32, 1024]`, finite and nonconstant values, and identical repeated inference for the first input. These checks verify a reproducible non-degenerate encoder execution; they do not establish forecast skill or a published benchmark score. Preserve the logs, checkpoint hash, data period, output hash, actual GPU, and SLURM resource report.

The documented full workflow uses a 2016 test shard. A 2017 NetCDF input encountered an HDF read error during preparation, so the successful run does not claim to cover 2017 or the full dataset. The original HDF traceback is not among the checked-in job logs; include it if available rather than implying that the failure log is preserved.

To independently inspect the saved NPZ and compare it against a second run, preserve both run directories and execute:

```bash
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/validate_embeddings.py" \
  "$HOME/scratch/climax-reproduction/runs/embeddings-6209978/climax-embeddings.npz" \
  --metrics "$HOME/scratch/climax-reproduction/runs/embeddings-6209978/metrics.json" \
  --compare "$HOME/scratch/climax-reproduction/runs/embeddings-<repeat-job-id>/climax-embeddings.npz"
```

This checks each numeric array for finite values, confirms the primary archive's SHA-256 and embedding dimensions against `metrics.json`, and compares all matching arrays using `rtol=1e-5` and `atol=1e-6`. The saved logs for jobs `6209971` ([log](workflows/climax-global-forecast/logs/climax-embed-6209971.out)) and `6209978` ([log](workflows/climax-global-forecast/logs/climax-embed-6209978.out)) record the same SHA-256 for their independently generated embedding archives (`7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158`), establishing byte-for-byte agreement for those outputs.

### Recorded PACE run

The frozen-encoder workflow completed successfully as SLURM job `6209978` (`climax-embed`), state `COMPLETED`, exit code `0:0`, on the `ice-gpu` partition, account `coc`, cluster ICE. The job ran on node `atl1-1-02-009-32-0` with one Tesla V100-PCIE-16GB GPU and 4 CPUs. SLURM reported 32 GiB of host memory for this job. The full stdout evidence is [`climax-embed-6209978.out`](workflows/climax-global-forecast/logs/climax-embed-6209978.out).

Observed validation: strict checkpoint loading completed; spatial embedding shape `(32, 512, 1024)`; pooled embedding shape `(32, 1024)`; all values finite; representation nonconstant; repeated first-sample inference deterministic. The output is in `$HOME/scratch/climax-reproduction/runs/embeddings-6209978/`. The recorded output hash and input/checkpoint provenance are in the linked SLURM log.

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

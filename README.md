# ClimaX inference on PACE ICE

This repository supports two ClimaX workflows on ERA5 geopotential at 500 hPa: frozen pretrained embeddings and 72-hour single-variable forecast prediction. The embedding workflow does not forecast weather. The forecast workflow fine-tunes the pinned upstream GlobalForecast model on `geopotential_500`; it accepts either the pressure-level `geopotential_5.625deg` archive or the direct single-level `geopotential_500_5.625deg` archive.

For detailed commands, troubleshooting, cleanup, and the optional full-variable forecast experiment, see the [workflow guide](workflows/climax-global-forecast/README.md).

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

Note that you can find your pace account using:

```bash
sacctmgr show associations user=$USER
```

Start an allocation using your assigned account and a currently available GPU partition:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" --partition=ice-gpu --gres=gpu:v100:1 \
  --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```

The pinned PyTorch 1.12.1/CUDA 11.3.1 environment was verified on a V100. Request it explicitly: a generic GPU request may assign a newer architecture such as H200 that is incompatible with this stack. If V100 is unavailable to your account, ask PACE before changing the GPU type or software pins.

Inside the allocation:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
bash scripts/setup_env.sh
conda activate "$HOME/scratch/.conda/envs/climax-inference"
python --version
```

The helper verifies PyTorch imports and runs a CUDA tensor smoke test using the environment's absolute Python path, then prints the Python, PyTorch, CUDA, and GPU versions. This avoids accidentally checking a different Python interpreter. It prefers Mamba, falls back to Conda with a notice, pins pip 24.0 and MKL 2024.0, and records Conda/pip inventories under `workflows/climax-global-forecast/execution/`. Full pins are in [`environment.yml`](workflows/climax-global-forecast/environment.yml). To rerun the detailed import/GPU diagnostics manually, use `"$HOME/scratch/.conda/envs/climax-inference/bin/python" scripts/check_environment.py --require-cuda`. To create the environment directly with Mamba instead of using the helper, follow the [Mamba alternative](workflows/climax-global-forecast/README.md#alternative-create-the-environment-directly-with-mamba) in the detailed guide.

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

Download source data from the [TUM WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) directly to ICE scratch storage. The embedding workflow needs `constants` plus the pressure-level `geopotential` archive. The geopotential-only forecast can use either the pressure-level `geopotential` archive or the direct `geopotential_500` archive; it does not need `constants`.

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

For the geopotential-only forecast, you may use this direct 500 hPa archive instead of the pressure-level `geopotential` archive:

```bash
curl -fL --retry 3 \
  -o geopotential_500.zip \
  'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fgeopotential_500'
```

These commands save the archives to scratch storage. Before starting a large transfer, check available storage:

```bash
du -h *.zip
df -h "$HOME/scratch"
```

If a download is interrupted, rerun the command to restart it. The commands above do not resume partial downloads.

#### 3. Verify the downloaded archives

```bash
if [ -f geopotential.zip ]; then
  file geopotential.zip
  unzip -t geopotential.zip
elif [ -f geopotential_500.zip ]; then
  file geopotential_500.zip
  unzip -t geopotential_500.zip
else
  echo "Download either geopotential.zip or geopotential_500.zip first." >&2
  exit 1
fi

if [ -f constants.zip ]; then
  file constants.zip
  unzip -t constants.zip
fi
```

Each downloaded archive should pass its ZIP integrity test. If using only `geopotential_500.zip`, test that archive and skip the following pressure-level staging steps; point `GEOPOTENTIAL_RAW_DIR` at its extracted contents in step 8.

#### 4. Extract the pressure-level outer archives (embedding route)

If you are using only `geopotential_500.zip` for forecasting, skip this and the following pressure-level staging steps; use the direct single-level extraction commands in step 5 and set `GEOPOTENTIAL_RAW_DIR` in step 8.

```bash
mkdir -p "$DOWNLOAD_DIR/constants" "$DOWNLOAD_DIR/geopotential"

if [ -f constants.zip ]; then
  unzip -n constants.zip -d "$DOWNLOAD_DIR"
fi
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

For the direct single-level archive, validate and extract the outer and nested ZIPs:

```bash
unzip -t geopotential_500.zip
unzip -n geopotential_500.zip -d "$DOWNLOAD_DIR"
unzip -n \
  "$DOWNLOAD_DIR/geopotential_500/geopotential_500_5.625deg.zip" \
  -d "$DOWNLOAD_DIR/geopotential_500"
find "$DOWNLOAD_DIR/geopotential_500" -maxdepth 3 -type f
```

Set `GEOPOTENTIAL_RAW_DIR="$DOWNLOAD_DIR/geopotential_500"` when using this source. For the pressure-level archive, use `"$DOWNLOAD_DIR/geopotential"`; the forecasting converter recursively finds its NetCDF files and extracts the 500 hPa slice.

#### 6. Stage the pressure-level files for embeddings

For the embedding workflow, copy the constants file to the canonical filename:

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

Forecast users can skip steps 6–7 and use `GEOPOTENTIAL_RAW_DIR` directly. If using the pressure-level files for forecasting, set it to `"$DOWNLOAD_DIR/geopotential"`; if using the single-level files, use `"$DOWNLOAD_DIR/geopotential_500"`.

#### 7. Verify the staged embedding dataset

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

#### 8. Preprocess the reduced input

Preprocessing is **required when starting from raw NetCDF files**. The embedding script and forecast script consume prepared `.npz` data, not NetCDF. The embedding path uses the pinned ClimaX converter with the `geopotential` pressure-level group and constants. The forecast path accepts either the pressure-level `geopotential_5.625deg` dataset or the direct single-level `geopotential_500_5.625deg` dataset; its dedicated converter extracts/normalizes the 500 hPa field and does not require constants or other variables.

Run preprocessing on an ICE CPU allocation. For example:

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
conda activate "$HOME/scratch/.conda/envs/climax-inference"

export CLIMAX_SOURCE="$HOME/scratch/climax-reproduction/ClimaX"
export PYTHONPATH="$CLIMAX_SOURCE/src:$PYTHONPATH"
```

For the forecast path, set `GEOPOTENTIAL_RAW_DIR` to the extracted contents of exactly one of the two geopotential datasets. Use a fresh output directory:

```bash
export GEOPOTENTIAL_RAW_DIR="$HOME/scratch/climax-reproduction/data/raw/geopotential_5.625deg"
export FORECAST_DATA="$HOME/scratch/climax-reproduction/data/geopotential_forecast_npz"
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/prepare_geopotential.py" \
  --input-dir "$GEOPOTENTIAL_RAW_DIR" \
  --output-dir "$FORECAST_DATA" \
  --train-start-year 2013 \
  --val-start-year 2015 \
  --test-start-year 2016 \
  --end-year 2017 \
  --num-shards 8
```

For the direct single-level archive, point `GEOPOTENTIAL_RAW_DIR` at its extracted folder instead, e.g. `$HOME/scratch/climax-reproduction/data/raw/geopotential_500_5.625deg`. The converter accepts pressure-level NetCDF fields containing the 500 hPa level and single-level `z`, `geopotential`, or `geopotential_500` fields. It writes one-variable shards, training normalization statistics, split climatologies, and coordinates. The 2013–2014 training, 2015 validation, 2016 test split is a pilot; choose years present in the downloaded source.

For the embedding path, use the pinned upstream converter:

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

Before using the prepared dataset, verify that it contains the expected split shards, normalization statistics, coordinate arrays, and `geopotential_500` variable. Forecasting additionally requires a fine-tuned GlobalForecast Lightning checkpoint; the pretrained `5.625deg.ckpt` alone is not a forecast model.

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

### 7. Fine-tune and run 72-hour geopotential forecasts

The prepared one-variable dataset from step 4 can train the upstream ClimaX GlobalForecast model to predict `geopotential_500` 72 hours ahead. From the workflow directory, configure the prepared dataset and submit the fine-tuning job:

```bash
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
export ERA5_NPZ="$HOME/scratch/climax-reproduction/data/geopotential_forecast_npz"
export TRAIN_DIR="$HOME/scratch/climax-reproduction/runs/geopotential-forecast"
export FORECAST_VARIABLE=geopotential_500
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/finetune.sbatch
squeue --me
```

The batch script starts from the pinned pretrained checkpoint and invokes the original ClimaX training code, overriding its input and target lists to the single geopotential field. Use the validation-selected `epoch_*.ckpt` from `"$TRAIN_DIR/checkpoints"` for inference. Alternatively, you may provide a fine-tuned Lightning checkpoint from the original ClimaX repository if it loads strictly with the pinned GlobalForecast architecture. Do not pass the pretrained `5.625deg.ckpt` as the fine-tuned forecast checkpoint.

Submit prediction on the held-out test split:

```bash
export OUTPUT_DIR="$HOME/scratch/climax-reproduction/runs/geopotential-predictions"
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/infer.sbatch \
  "$TRAIN_DIR/checkpoints/epoch_XXX.ckpt"
squeue --me
```

The output NPZ contains 72-hour predictions, matching ERA5 targets, persistence and climatology baselines, and coordinates. `metrics.json` reports finite-value checks and latitude-weighted RMSE for the single output variable; expected prediction shape is `[32, 1, 32, 64]`. Compare against both baselines and report the observed result, not a presumed model improvement. The previously recorded PACE validation applies to the embedding workflow only; run these new preprocessing, training, and prediction commands on PACE and retain their logs and metrics before claiming the forecast path is empirically reproduced.

### Recorded PACE run

The frozen-encoder workflow completed successfully as SLURM job `6209978` (`climax-embed`), state `COMPLETED`, exit code `0:0`, on the `ice-gpu` partition, account `coc`, cluster ICE. The job ran on node `atl1-1-02-009-32-0` with one Tesla V100-PCIE-16GB GPU and 4 CPUs. SLURM reported 32 GiB of host memory for this job. The full stdout evidence is [`climax-embed-6209978.out`](workflows/climax-global-forecast/logs/climax-embed-6209978.out).

Observed validation: strict checkpoint loading completed; spatial embedding shape `(32, 512, 1024)`; pooled embedding shape `(32, 1024)`; all values finite; representation nonconstant; repeated first-sample inference deterministic. The output is in `$HOME/scratch/climax-reproduction/runs/embeddings-6209978/`. The recorded output hash and input/checkpoint provenance are in the linked SLURM log.

## Scope and limitations

- The recorded no-fine-tuning run extracts representations; it does not produce future weather fields.
- The input is intentionally reduced to `geopotential_500` and a short ERA5 period to reduce data and compute requirements.
- Representation non-degeneracy and deterministic repeat checks are technical validation, not a downstream scientific skill metric.
- The reduced geopotential forecast workflow fine-tunes the upstream ClimaX forecaster on one input/target variable. The original five-output forecast experiment remains optional and requires the full forecast input variables.
- The new forecast path has not yet been executed on PACE; the existing successful PACE log and metrics validate embeddings only.
- Do not commit large data, checkpoints, or private PACE logs unless repository policy permits. Keep run evidence in an approved location.

## Supporting documentation

- [Detailed workflow, troubleshooting, and cleanup](workflows/climax-global-forecast/README.md)
- [PACE ICE documentation](https://docs.pace.gatech.edu/ice/)
- [Pinned ClimaX pretraining configuration](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/pretrain_climax.yaml)
- [Pinned ClimaX usage guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) and [WeatherBench README](https://github.com/pangeo-data/WeatherBench)

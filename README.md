# ClimaX inference on PACE ICE

This repository supports frozen pretrained embeddings, quick 72-hour baseline forecasts, and optional fine-tuning of the ClimaX GlobalForecast model on ERA5 geopotential at 500 hPa. Embeddings are representations, not weather predictions. Persistence and training-climatology baselines provide a fast forecast-pipeline check without model training. The optional model forecast uses a fine-tuned GlobalForecast checkpoint and accepts either the pressure-level `geopotential_5.625deg` archive or the direct single-level `geopotential_500_5.625deg` archive.

**Use this README as the single step-by-step runbook** for setup, downloads, preprocessing, embedding extraction, baseline forecasts, and optional model fine-tuning. The baseline forecast needs no GPU or training and is the quick route for checking prediction generation. The full data download and preprocessing may take longer than an hour. The [supplementary workflow reference](workflows/climax-global-forecast/README.md) contains technical background, historical run evidence, and optional material; you do not need to switch to it to complete the standard workflow.

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

The previous successful jobs in this workflow used account `coc`. Confirm your assigned account first; do not enter the literal placeholder `<YOUR_PACE_ACCOUNT>` in a command. If your account is still `coc`, start an allocation from the login shell with:

```bash
srun --account=coc --partition=ice-gpu --gres=gpu:v100:1 \
  --cpus-per-task=4 --mem=16G --time=01:00:00 --pty /bin/bash -l
```

If your assigned account differs, replace `coc` with the account listed by `sacctmgr show associations user="$USER"`. `/bin/bash -l` explicitly starts a login Bash on the allocated node. If Slurm still reports `execve():  : No such file or directory`, the allocation could not launch its requested shell; this is separate from Python package installation. Retry with `srun --verbose` and retain the full command and output for PACE support.

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

The helper verifies PyTorch, NumPy, xarray, and ClimaX dependency imports and runs a CUDA tensor smoke test using the environment's absolute Python path, then prints the Python, PyTorch, CUDA, and GPU versions. This avoids accidentally checking a different Python interpreter. It prefers Mamba, falls back to Conda with a notice, pins pip 24.0 and MKL 2024.0, and records Conda/pip inventories under `workflows/climax-global-forecast/execution/`. Full pins are in [`environment.yml`](workflows/climax-global-forecast/environment.yml). To rerun the detailed import/GPU diagnostics manually, use `"$HOME/scratch/.conda/envs/climax-inference/bin/python" scripts/check_environment.py --require-cuda`.

`environment.yml` is the dependency source of truth: it already includes pinned `numpy` and `xarray`. Do not manually add Conda-installed packages to the generated `execution/pip-freeze.txt`; that file is an inventory captured by `setup_env.sh`, not an installation manifest. If an import is missing, rerun setup and verify the exact environment:

```bash
bash scripts/setup_env.sh
"$HOME/scratch/.conda/envs/climax-inference/bin/python" -c \
  'import numpy, xarray; print("numpy", numpy.__version__); print("xarray", xarray.__version__)'
"$HOME/scratch/.conda/envs/climax-inference/bin/python" scripts/check_environment.py --require-cuda
```

Run the import command inside the GPU allocation after `conda activate "$HOME/scratch/.conda/envs/climax-inference"`. If the absolute-path Python command imports the modules but plain `python` does not, the active shell has the wrong interpreter; reactivate the documented environment rather than changing the package list.

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

#### 1. Set the data paths and create the directories

```bash
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export DOWNLOAD_DIR="$PROJECT_DIR/downloads"
export DATA_DIR="$PROJECT_DIR/data/5.625deg"
export PREPROCESS_DATA="$PROJECT_DIR/data/5.625deg_2013_2016"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
export FORECAST_DATA="$PROJECT_DIR/data/geopotential_forecast_npz"

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR/geopotential" "$PROJECT_DIR/checkpoints"
cd "$DOWNLOAD_DIR"
```

These names distinguish the stages of the workflow: `downloads/` holds downloaded and extracted source archives; `data/5.625deg/` holds staged raw NetCDF files; `data/geopotential_npz_2016_test/` is the prepared embedding input; and `data/geopotential_forecast_npz/` is the separately prepared forecasting input. The two `*_npz` directories are generated by preprocessing and are not download locations.

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

Each downloaded archive should pass its ZIP integrity test. If using only `geopotential_500.zip`, test that archive and skip the following pressure-level staging steps; use its extracted directory, `"$DOWNLOAD_DIR/geopotential_500"`, as `GEOPOTENTIAL_RAW_DIR` in step 8.

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

#### 5. Extract the nested geopotential archive you downloaded

For the pressure-level `geopotential.zip` archive:

```bash
unzip -n \
  "$DOWNLOAD_DIR/geopotential/geopotential_5.625deg.zip" \
  -d "$DOWNLOAD_DIR/geopotential"
find "$DOWNLOAD_DIR/constants" -maxdepth 3 -type f
find "$DOWNLOAD_DIR/geopotential" -maxdepth 3 -type f
```

If the nested archive contains additional ZIP files rather than individual NetCDF files, inspect those archives before proceeding.

For the direct single-level `geopotential_500.zip` archive, use this block instead:

```bash
unzip -t geopotential_500.zip
unzip -n geopotential_500.zip -d "$DOWNLOAD_DIR"
unzip -n \
  "$DOWNLOAD_DIR/geopotential_500/geopotential_500_5.625deg.zip" \
  -d "$DOWNLOAD_DIR/geopotential_500"
find "$DOWNLOAD_DIR/geopotential_500" -maxdepth 3 -type f
```

Use `"$DOWNLOAD_DIR/geopotential_500"` as `GEOPOTENTIAL_RAW_DIR` for this single-level source. For the pressure-level archive, use `"$DOWNLOAD_DIR/geopotential"`; the forecasting converter recursively finds its NetCDF files and extracts the 500 hPa slice.

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

For forecasting, use the year-filtered preprocessing subset described in step 8 below, not the full pressure-level directory. If using the optional single-level archive instead, also filter it to 2013–2016 before passing it to the forecast converter.

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

export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export DOWNLOAD_DIR="$PROJECT_DIR/downloads"
export DATA_DIR="$PROJECT_DIR/data/5.625deg"
export PREPROCESS_DATA="$PROJECT_DIR/data/5.625deg_2013_2016"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
export FORECAST_DATA="$PROJECT_DIR/data/geopotential_forecast_npz"
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
export PYTHONPATH="$CLIMAX_SOURCE/src:$PYTHONPATH"
```

Both converters open NetCDF files before selecting the requested years. Do not point them at the full staged directory: it may contain a 2017 file that has produced an HDF read error. Instead, make an isolated raw-input subset containing only 2013–2016; the original download and `data/5.625deg/` remain unchanged. The commands below use the pressure-level archive, which supports both workflows:

```bash
export PREPROCESS_DATA="$PROJECT_DIR/data/5.625deg_2013_2016"
mkdir -p "$PREPROCESS_DATA/geopotential"
if [ -f "$DATA_DIR/constants.nc" ]; then
  cp -f "$DATA_DIR/constants.nc" "$PREPROCESS_DATA/constants.nc"
else
  echo "constants.nc is absent; forecast preprocessing can proceed, but embedding preprocessing requires it." >&2
fi

for year in 2013 2014 2015 2016; do
  source_file="$DATA_DIR/geopotential/geopotential_${year}_5.625deg.nc"
  if [ ! -s "$source_file" ]; then
    echo "Missing or empty required input: $source_file" >&2
    exit 1
  fi
  cp -f "$source_file" "$PREPROCESS_DATA/geopotential/"
done

find "$PREPROCESS_DATA" -maxdepth 2 -type f -name '*.nc' -print
```

Verify the listing contains only the four intended yearly geopotential files and no 2017 file. `constants.nc` should also appear when you downloaded it for embedding preprocessing; it is optional for the forecast-only converter. If your extracted filenames differ, list `"$DATA_DIR/geopotential"` and adjust the `source_file` pattern to select one file per year; do not copy the full directory into the preprocessing subset.

For the forecast path, use the isolated geopotential directory as input. Do not use `data/geopotential_npz` or `data/geopotential_npz_2016_test` as converter input; those names refer to prepared data, not raw NetCDF files. The converter writes the one-variable forecast data to the dedicated output directory:

```bash
export GEOPOTENTIAL_RAW_DIR="$PREPROCESS_DATA/geopotential"
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/prepare_geopotential.py" \
  --input-dir "$GEOPOTENTIAL_RAW_DIR" \
  --output-dir "$FORECAST_DATA" \
  --train-start-year 2013 \
  --val-start-year 2015 \
  --test-start-year 2016 \
  --end-year 2017 \
  --num-shards 8
```

If you use the optional direct single-level archive instead, first copy only its 2013–2016 NetCDF files from `"$DOWNLOAD_DIR/geopotential_500"` into a separate directory and set `GEOPOTENTIAL_RAW_DIR` to that directory. Do not point the converter directly at the complete extracted archive if it includes the unreadable 2017 file. The converter accepts pressure-level NetCDF fields containing the 500 hPa level and single-level `z`, `geopotential`, or `geopotential_500` fields. It writes one-variable shards, training normalization statistics, split climatologies, and coordinates to `"$FORECAST_DATA"`. The 2013–2014 training, 2015 validation, 2016 test split is a pilot; choose years present in the downloaded source.

For the embedding path, use the pinned upstream converter:

```bash
python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$PREPROCESS_DATA" \
  --save_dir "$EMBEDDING_DATA" \
  --start_train_year 2013 \
  --start_val_year 2015 \
  --start_test_year 2016 \
  --end_year 2017 \
  --num_shards 8 \
  --variables geopotential
```

Here, `end_year` is exclusive: training uses 2013–2014, validation uses 2015, and the test split uses 2016. The isolated `PREPROCESS_DATA` directory ensures neither converter attempts to open the 2017 file. This matches the successful `2016_0.npz` test shard; do not represent the run as covering 2017 or as a full-data benchmark.

The embedding job reads from `"$EMBEDDING_DATA"` (`data/geopotential_npz_2016_test`). A directory named `data/geopotential_npz` is not created by the commands above and was the path used by an earlier unsuccessful attempt; do not point the embedding job at it. If you already have that directory, retain it, but use the explicitly generated `geopotential_npz_2016_test` directory for this documented test workflow.

Before using the prepared dataset, verify that it contains the expected split shards, normalization statistics, coordinate arrays, and `geopotential_500` variable. The baseline forecast below runs directly from this data. A ClimaX model forecast additionally requires a fine-tuned GlobalForecast Lightning checkpoint; the pretrained `5.625deg.ckpt` alone is not a trained forecast model.

### 5. Generate a quick 72-hour baseline forecast

This CPU-only step produces genuine test-period forecast comparisons without training: persistence carries the initial field forward, while the climatology baseline uses the training-split mean. It writes numerical predictions, latitude-weighted RMSE metrics against the observed 72-hour targets, and an SVG map comparison. This verifies the prepared forecast data path and produces a visual result; it is **not a ClimaX prediction**, and the scores are not guaranteed to indicate useful forecast skill.

Run this in the initialized environment after step 4:

```bash
export BASELINE_DIR="$PROJECT_DIR/runs/geopotential-baseline"
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/baseline_forecast.py" \
  --data-dir "$FORECAST_DATA" \
  --output-dir "$BASELINE_DIR" \
  --variable geopotential_500 \
  --split test \
  --lead-hours 72 \
  --num-samples 32 \
  --sample-stride 24
```

Inspect the results:

```bash
cat "$BASELINE_DIR/metrics.json"
ls -lh "$BASELINE_DIR"
```

The output includes `baseline-forecast.npz`, `metrics.json`, and `baseline-forecast.svg`. The SVG shows the first input, observed 72-hour target, persistence forecast, and training climatology on one shared color scale. Open the SVG in a browser or copy it to your computer. The metrics aggregate 32 daily-spaced examples from the test shard. A lower latitude-weighted RMSE is better; compare each baseline to the same target period. This baseline path is the quick forecast demonstration and does not require the fine-tuning section later in the README.

### 6. Run frozen pretrained embedding inference

Return to the login shell, initialize Miniforge there, and submit:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

The job requests one GPU, 4 CPUs, 32 GiB RAM, and one hour. It strictly loads the pinned checkpoint into the upstream pretraining module, encodes 32 daily-spaced samples from the first sorted test shard, and writes outputs to `$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>/`.

To run interactively instead, use the same arguments as `slurm/embed.sbatch` with `python scripts/embed.py` from the workflow directory. The [embedding script](workflows/climax-global-forecast/scripts/embed.py) and its `--help` output list the supported arguments.

### 7. Verify and preserve embedding evidence

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

Job `6209978` is a recorded successful reference run: it completed with exit code `0:0`, passed the embedding checks, and produced the documented output archive. To compare your run against that independent reference, set `JOB_ID` to the job you just completed and `REFERENCE_JOB_ID` to `6209978`:

```bash
JOB_ID="<your-successful-job-id>"
REFERENCE_JOB_ID=6209978
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/validate_embeddings.py" \
  "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID/climax-embeddings.npz" \
  --metrics "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID/metrics.json" \
  --compare "$HOME/scratch/climax-reproduction/runs/embeddings-$REFERENCE_JOB_ID/climax-embeddings.npz"
```

Here, `$JOB_ID` is replaced by your new successful run's ID; `$REFERENCE_JOB_ID` remains `6209978`. Both output archives must exist on the filesystem where you run the validator. If the `6209978` results are no longer available on your scratch storage, either restore that archive from your retained copy or compare with another successful job by changing `REFERENCE_JOB_ID`. The validator checks each numeric array for finite values, confirms the new run's SHA-256 and dimensions against its metrics, and compares matching arrays using `rtol=1e-5` and `atol=1e-6`. Historical logs show jobs `6209971` and `6209978` produced the same output SHA-256.

### 8. Optional: fine-tune ClimaX and generate model forecasts

The prepared one-variable dataset from step 4 can train the upstream ClimaX GlobalForecast model to predict `geopotential_500` 72 hours ahead. This section is optional: step 5 already produces forecast predictions and plots without model training.

#### Optional quick adaptation

For an exploratory model-vs-baseline comparison, the batch script supports a short training run with an explicit one-hour wall-time cap. It limits training to 100 batches/optimizer steps and validation/test to 8 batches each; it also shortens the learning-rate warmup so those few updates are meaningful. This is a smoke-scale adaptation, not the paper's full training recipe, and it may not improve RMSE over persistence or climatology. Compare the observed metrics rather than assuming improvement. The default remains the full-training configuration unless `QUICK_TUNE_STEPS` is exported.

From the PACE login shell, set the prepared data paths and test that the request is valid:

```bash
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export ERA5_NPZ="$PROJECT_DIR/data/geopotential_forecast_npz"
export TRAIN_DIR="$PROJECT_DIR/runs/geopotential-forecast-quick"
export FORECAST_VARIABLE=geopotential_500
export QUICK_TUNE_STEPS=100
mkdir -p logs
sbatch --test-only --time=01:00:00 --account=coc slurm/finetune.sbatch
```

Use your assigned account if it is not `coc`. If the test-only request succeeds, submit:

```bash
sbatch --time=01:00:00 --account=coc slurm/finetune.sbatch
squeue --me
```

When it completes, select the checkpoint written under `"$TRAIN_DIR/checkpoints"` (usually `epoch_*.ckpt`) and run inference on the same held-out test period:

```bash
export FINETUNED_CHECKPOINT="$TRAIN_DIR/checkpoints/epoch_XXX.ckpt"
export OUTPUT_DIR="$PROJECT_DIR/runs/geopotential-forecast-quick-predictions"
if [ ! -f "$FINETUNED_CHECKPOINT" ]; then
  echo "Replace epoch_XXX.ckpt with the checkpoint that exists under $TRAIN_DIR/checkpoints." >&2
  exit 1
fi
sbatch --time=01:00:00 --account=coc slurm/infer.sbatch "$FINETUNED_CHECKPOINT"
```

The inference job writes `climax-forecast.npz`, `metrics.json`, and `climax-forecast.svg`. The SVG shows the input, 72-hour target, ClimaX output, persistence, and training climatology on a shared scale. The full inference metrics compare all selected samples to both baselines. If the short run reaches its time limit before writing a validation-selected checkpoint, use the baseline-only result from step 5; do not treat an interrupted run as a completed ClimaX forecast.

#### Full fine-tuning (optional longer experiment)

The full training configuration uses up to 50 epochs and may need more than one hour. The `ice-gpu` partition details supplied for this workflow show `MaxTime=16:00:00` and `DefaultTime=01:00:00`: 16 hours is the partition maximum, while one hour is the default only when a job does not request a time. An account/QOS can still impose a lower limit, so check before submitting. Start this in a fresh shell or unset `QUICK_TUNE_STEPS` and use a separate output directory from the exploratory run.

From the PACE login shell (not inside an existing interactive `srun` allocation), inspect the ICE GPU partition and your assigned QOS/account limits:

```bash
scontrol show partition ice-gpu
sacctmgr show assoc where user="$USER" format=Account,QOS,MaxWall
```

If the association output lists a QOS, inspect its maximum wall time (replace the example QOS with the one reported for your account):

```bash
sacctmgr show qos "<YOUR_QOS>" format=Name,MaxWall
```

Use the smallest applicable limit: the partition maximum, the QOS maximum, or the account association maximum. Blank `MaxWall` fields do not establish that the account can use the full partition limit. If these fields are blank or unclear, test the 16-hour request with `sbatch --test-only`; if it is rejected, ask PACE support for the applicable account/QOS limit and choose a shorter time. Do not retry an invalid request unchanged. `DefaultTime=01:00:00` does not cap a job that explicitly requests a longer time. The scheduler message `Requested time limit is invalid` means the request was rejected before a job ID was assigned. A job shown by `squeue` with `NAME=bash` is a separate interactive allocation, not the rejected fine-tuning job.

The scheduler output you provided reports a partition maximum of 16 hours. Use that as the initial request, and let `--test-only` verify that your account/QOS allows it:

```bash
FINETUNE_TIME=16:00:00
unset QUICK_TUNE_STEPS
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export ERA5_NPZ="$PROJECT_DIR/data/geopotential_forecast_npz"
export TRAIN_DIR="$PROJECT_DIR/runs/geopotential-forecast"
export FORECAST_VARIABLE=geopotential_500
if [ ! -d "$ERA5_NPZ" ]; then
  echo "Forecast data not found: $ERA5_NPZ. Run the forecast preprocessing command in step 4 first." >&2
  exit 1
fi
mkdir -p logs
sbatch --test-only --time="$FINETUNE_TIME" --account=coc slurm/finetune.sbatch
```

Use your assigned account if it is not `coc`. `sbatch --test-only` validates the request without submitting or starting training. Only if it accepts the request, submit it:

```bash
sbatch --time="$FINETUNE_TIME" --account=coc slurm/finetune.sbatch
squeue --me
```

Confirm that `sbatch` prints a new job ID before using `squeue` or checking logs. The batch script starts from the pinned pretrained checkpoint and invokes the original ClimaX training code, overriding its input and target lists to the single geopotential field. Use the validation-selected `epoch_*.ckpt` from `"$TRAIN_DIR/checkpoints"` for inference. Alternatively, you may provide a fine-tuned Lightning checkpoint from the original ClimaX repository if it loads strictly with the pinned GlobalForecast architecture. Do not pass the pretrained `5.625deg.ckpt` as the fine-tuned forecast checkpoint.

The fine-tuning script uses Lightning's `single_device` strategy because the pinned Lightning 1.8.0 environment does not register the newer `auto` strategy name. This is appropriate for the one-GPU job requested above.

If training reaches the allowed wall time, check whether `"$TRAIN_DIR/checkpoints/last.ckpt"` was written. If present, resume from it by submitting the script with that checkpoint path as its final argument; this resumes training, it is not the checkpoint to use for final forecast inference. If your account accepts only a shorter time, set `FINETUNE_TIME` to that accepted duration before resuming:

```bash
sbatch --time="$FINETUNE_TIME" --account="<YOUR_PACE_ACCOUNT>" \
  slurm/finetune.sbatch "$TRAIN_DIR/checkpoints/last.ckpt"
```

Submit prediction on the held-out test split:

```bash
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export DOWNLOAD_DIR="$PROJECT_DIR/downloads"
export DATA_DIR="$PROJECT_DIR/data/5.625deg"
export EMBEDDING_DATA="$PROJECT_DIR/data/geopotential_npz_2016_test"
export FORECAST_DATA="$PROJECT_DIR/data/geopotential_forecast_npz"
export TRAIN_DIR="$PROJECT_DIR/runs/geopotential-forecast"
export OUTPUT_DIR="$PROJECT_DIR/runs/geopotential-predictions"
export ERA5_NPZ="$FORECAST_DATA"
export FORECAST_VARIABLE=geopotential_500
export FINETUNED_CHECKPOINT="$TRAIN_DIR/checkpoints/epoch_XXX.ckpt"
if [ ! -f "$FINETUNED_CHECKPOINT" ]; then
  echo "Set FINETUNED_CHECKPOINT to the validation-selected epoch checkpoint under $TRAIN_DIR/checkpoints." >&2
  exit 1
fi
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/infer.sbatch "$FINETUNED_CHECKPOINT"
squeue --me
```

The output NPZ contains 72-hour predictions, matching ERA5 targets, persistence and training-climatology baselines, and coordinates. `metrics.json` reports finite-value checks and latitude-weighted RMSE for the single output variable; expected prediction shape is `[32, 1, 32, 64]`. `climax-forecast.svg` visualizes the first input, target, model output, and baselines on a shared scale. The climatology baseline uses the training split, not the held-out test climatology. Compare against both baselines and report the observed result, not a presumed model improvement. The previously recorded PACE validation applies to the embedding workflow only; run these new preprocessing, training, and prediction commands on PACE and retain their logs and metrics before claiming the forecast path is empirically reproduced.

### Recorded PACE run

The frozen-encoder workflow completed successfully as SLURM job `6209978` (`climax-embed`), state `COMPLETED`, exit code `0:0`, on the `ice-gpu` partition, account `coc`, cluster ICE. The job ran on node `atl1-1-02-009-32-0` with one Tesla V100-PCIE-16GB GPU and 4 CPUs. SLURM reported 32 GiB of host memory for this job. The full stdout evidence is [`climax-embed-6209978.out`](workflows/climax-global-forecast/logs/climax-embed-6209978.out).

Observed validation: strict checkpoint loading completed; spatial embedding shape `(32, 512, 1024)`; pooled embedding shape `(32, 1024)`; all values finite; representation nonconstant; repeated first-sample inference deterministic. The output is in `$HOME/scratch/climax-reproduction/runs/embeddings-6209978/`. The recorded output hash and input/checkpoint provenance are in the linked SLURM log.

## Scope and limitations

- The recorded no-fine-tuning run extracts representations; it does not produce future weather fields.
- The input is intentionally reduced to `geopotential_500` and a short ERA5 period to reduce data and compute requirements.
- Representation non-degeneracy and deterministic repeat checks are technical validation, not a downstream scientific skill metric.
- The reduced geopotential forecast workflow fine-tunes the upstream ClimaX forecaster on one input/target variable. The original five-output forecast experiment remains optional and requires the full forecast input variables.
- The fast persistence/climatology baseline is not a pretrained ClimaX forecast and does not establish model skill. The short optional fine-tuning run is exploratory and may not outperform those baselines.
- The new forecast path has not yet been executed on PACE; the existing successful PACE log and metrics validate embeddings only.
- Do not commit large data, checkpoints, or private PACE logs unless repository policy permits. Keep run evidence in an approved location.

## Troubleshooting

- **Fine-tuning submission says `Requested time limit is invalid`:** The script requests 16 hours, matching the `ice-gpu` partition maximum reported for this workflow. Your account/QOS may allow less. Follow the wall-time preflight, use `sbatch --test-only` to check the chosen `--time`, and lower `FINETUNE_TIME` if needed. If training reaches its accepted limit, resume only if `"$TRAIN_DIR/checkpoints/last.ckpt"` exists.
- **Job shows `FAILED`, or `metrics.json` is missing:** Check the job's actual `.err` and `.out` logs under `workflows/climax-global-forecast/logs/`. A missing output file alone does not identify the cause. `sacct`'s `extern` row completing does not mean the batch task succeeded; check the main and `.batch` rows and their exit codes.
- **`ModuleNotFoundError: No module named 'torch'`:** Load Miniforge, activate `"$HOME/scratch/.conda/envs/climax-inference"`, then run `"$HOME/scratch/.conda/envs/climax-inference/bin/python" scripts/check_environment.py --require-cuda`. If the check fails, run `bash scripts/setup_env.sh` from the workflow directory and keep the error output. The helper uses Mamba when available and falls back to Conda.
- **`ModuleNotFoundError: No module named 'xarray'` or `numpy`:** Both are already pinned in `environment.yml`; do not hand-edit `execution/pip-freeze.txt`. Run `bash scripts/setup_env.sh`, activate the exact environment, and verify with the absolute-path Python import command in step 2. If that still fails, include the complete setup output and `python -c 'import sys; print(sys.executable)'` output when asking for help.
- **Fine-tuning fails with `ModuleNotFoundError: No module named 'climax'`:** The ClimaX source checkout is separate from the Python environment. The updated `slurm/finetune.sbatch` adds `"$CLIMAX_SOURCE/src"` to `PYTHONPATH` and checks `import climax` before starting training. Copying this fix uses `scp`, which must be run on your **local computer**, not in a shell whose prompt is on PACE (for example, `pkathalkar3@login-ice-gnr-1`). Open a local terminal, `cd` to the local clone of `climax-inference`, and confirm the source file exists:

  ```bash
  cd "/path/to/your/local/climax-inference"
  ls workflows/climax-global-forecast/slurm/finetune.sbatch
  scp workflows/climax-global-forecast/slurm/finetune.sbatch \
    "<YOUR_PACE_USERNAME>@login-ice.pace.gatech.edu:~/scratch/climax-inference/workflows/climax-global-forecast/slurm/finetune.sbatch"
  ```

  Replace `/path/to/your/local/climax-inference` and `<YOUR_PACE_USERNAME>` with your local clone path and PACE username. Keep the remote destination on one line. The error `scp: stat local ...: No such file or directory` means the local source file was not found from the current directory; it commonly happens when `scp` was run on PACE instead of locally. After copying, return to a PACE shell and confirm the remote script now contains `PYTHONPATH` and the `import climax` check:

  ```bash
  grep -n 'PYTHONPATH\|import climax' \
    "$HOME/scratch/climax-inference/workflows/climax-global-forecast/slurm/finetune.sbatch"
  ```

  If `grep` prints nothing, the PACE copy is still old. Once the lines appear, rerun the `sbatch --test-only` time check and submit a new job; do not recreate the environment or rerun preprocessing for this import-path error. The job output should print `ClimaX package:` with a path inside `"$PROJECT_DIR/ClimaX/src/climax"` before training starts.
- **Fine-tuning fails with `KeyError: "'auto' not found in registry"`:** The job requested `--trainer.strategy=auto`, but the pinned PyTorch Lightning 1.8.0 registry shown in the traceback does not support that strategy. The updated `slurm/finetune.sbatch` uses the registered `single_device` strategy for this one-GPU job. Copy the updated script from your local computer using the instructions above, verify it contains `--trainer.strategy=single_device`, then resubmit. The Mamba initialization notice, Lightning deprecation warnings, and Lightning's warning that `srun` is not used are non-fatal messages; in the shown traceback, the strategy `KeyError` is what stopped training.
- **`libxml2_deactivate.sh` reports `xml_catalog_files_libxml2: unbound variable`:** This means an old job script is running `module purge` or `module load miniforge` while Bash `nounset` is enabled. The fixed `finetune.sbatch` turns `nounset` off before those module commands and restores it after Conda activation. If the fix is only in your local checkout, copy the script using the local-computer `scp` instructions in the preceding troubleshooting item.

  On PACE, verify `set +u` appears before `module purge` and `set -u` appears after `conda activate`:

  ```bash
  cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
  sed -n '20,42p' slurm/finetune.sbatch
  ```

  Then rerun the `sbatch --test-only` time-limit check and submit the fine-tuning job. No environment, checkpoint, or dataset recreation is needed. Do not edit the installed PACE Miniforge scripts.

- **CUDA is unavailable or unsupported:** Run the checker inside the requested GPU allocation, confirm the node with `nvidia-smi`, and use the documented V100-compatible environment. Do not treat CPU fallback as successful GPU inference.

## Cleanup

After saving any logs, metrics, and artifacts needed for your report, inspect exact paths before optional deletion. Do not remove the downloaded source data or workflow checkout unless you intend to download/clone them again.

```bash
ls -ld "$HOME/scratch/climax-reproduction/runs/"*
ls -ld "$HOME/scratch/climax-reproduction/checkpoints"
```

The [supplementary workflow reference](workflows/climax-global-forecast/README.md) is not needed to execute the standard recipe. It is retained for technical background and historical notes; the [assignment evidence record](workflows/climax-global-forecast/A2DETAILS.md) and [PR summary](PULL_REQUEST.md) provide additional provenance and results.

- [PACE ICE documentation](https://docs.pace.gatech.edu/ice/)
- [Pinned ClimaX pretraining configuration](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/pretrain_climax.yaml)
- [Pinned ClimaX usage guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) and [WeatherBench README](https://github.com/pangeo-data/WeatherBench)

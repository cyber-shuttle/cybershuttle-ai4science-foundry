# ClimaX global forecasting on Georgia Tech PACE ICE

This guide takes a student from an ICE login through environment setup, data preparation, forecast fine-tuning, a held-out ERA5 evaluation, saved predictions, and cleanup. It is the detailed companion to the repository's [top-level reproduction guide](../../README.md).

## What this run reproduces

This is a **global forecast fine-tuning and inference workflow**, not an attempt to treat ClimaX pretraining weights alone as a validated forecaster:

1. Initialize Microsoft's ClimaX global-forecast model with the public 5.625° pretrained checkpoint.
2. Fine-tune the forecast task on ERA5 using the upstream ClimaX trainer and configuration.
3. Run inference using a saved fine-tuned forecast checkpoint.
4. Check a sample set against the observed ERA5 target, persistence, and test climatology, and retain the trainer's validation/test metrics.

The standard target is the upstream 72-hour forecast configuration. The upstream model documentation demonstrates 50 fine-tuning epochs; the supplied SLURM template uses that setting, one GPU, batch size 4, and mixed precision. These resources are starting requests, not measured performance guarantees. Full ERA5 fine-tuning can take substantial GPU time.

## 1. Provenance and fixed artifacts

| Artifact | Source / identifier | Use |
|---|---|---|
| Model implementation | [Microsoft/ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c) | Forecast architecture, upstream trainer, data preprocessing |
| Initialization weights | [`tungnd/climax`, revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, `5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt) | Pretraining initialization for ERA5 global-forecast fine-tuning |
| Checkpoint SHA-256 | `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43` | Verify the downloaded initial weights |
| Weather inputs | [WeatherBench ERA5 archive linked by upstream ClimaX](https://dataserv.ub.tum.de/index.php/s/m1524895) | ERA5 predictors, targets, split climatology |
| Forecast definition | Pinned ClimaX [`global_forecast_climax.yaml`](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml) | Grid, inputs, outputs, lead time, model dimensions |

The checkpoint is **not** an ERA5-fine-tuned global-forecast checkpoint. The fine-tuning job adapts it for forecasting; the subsequent inference command must point to the saved forecasting checkpoint, not the initial `5.625deg.ckpt`.

## 2. PACE access and storage

You need a Georgia Tech PACE ICE account, permission to use a GPU allocation, an allocation/account name, and sufficient scratch storage for the environment, ClimaX source, ERA5 data, checkpoints, and run output. Do not assume a particular project/account name. Replace `<YOUR_PACE_ACCOUNT>` below with an account assigned to you by PACE; `coc` may be appropriate for some users but is not universal.

Connect using the login host in the [PACE ICE documentation](https://docs.pace.gatech.edu/ice/) and open an interactive GPU shell. This example follows the `ice-gpu` partition and SLURM request style documented by PACE users; verify the current partition and account requirements for your allocation:

```bash
export GT_USERNAME="<YOUR_GT_USERNAME>"
ssh "$GT_USERNAME@login-ice.pace.gatech.edu"
export PACE_ACCOUNT="<YOUR_PACE_ACCOUNT>"
srun --account="$PACE_ACCOUNT" --partition=ice-gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

The inference and fine-tuning batch scripts request one `ice-gpu` GPU. Fine-tuning asks for 8 CPUs, 64 GiB RAM, and 24 hours; inference asks for 4 CPUs, 32 GiB RAM, and 1 hour. Requests may need adjustment to your account's limits and the queue. Record the *actual* GPU model, allocated resources, elapsed time, and peak memory; no run has yet been performed from this repository.

Use PACE scratch for caches and large files. The commands below use `$HOME/scratch`; confirm the location/quota for your account. Redirecting Conda, pip, temporary, Hugging Face, and PyTorch caches avoids filling a small home quota.

## 3. Clone this workflow and the pinned model source

Run on ICE. The repository's `climax-inference` branch is the destination for this assignment:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$HOME/scratch"
export WORKFLOW_REPO="$HOME/scratch/climax-inference"
git clone --branch climax-inference \
  https://github.com/purvikathalkar/climax-inference.git "$WORKFLOW_REPO"
export WORKFLOW_DIR="$WORKFLOW_REPO/workflows/climax-global-forecast"
cd "$WORKFLOW_DIR"

export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export ENV_PREFIX="$HOME/scratch/.conda/envs/climax-inference"
mkdir -p "$PROJECT_DIR"
git clone https://github.com/microsoft/ClimaX.git "$PROJECT_DIR/ClimaX"
git -C "$PROJECT_DIR/ClimaX" checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
git -C "$CLIMAX_SOURCE" rev-parse HEAD | tee "$WORKFLOW_DIR/execution/climax-source-commit.txt"
```

If `git clone --branch climax-inference` says the branch does not exist, the branch has not yet been pushed to GitHub. Use the repository's published assignment branch once available; do not silently clone `main` and claim it is the reproduction revision.

## 4. Load Conda correctly and create the environment

On PACE, Conda is supplied through the Miniforge environment module. A new login shell does not necessarily have a `conda` command on `PATH`; this is why running `conda env create` immediately can return `-bash: conda: command not found`.

In each fresh shell that uses Conda, initialize it in this order:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
command -v conda
conda --version
```

Create the environment **inside a GPU allocation**, so CUDA visibility and the allocated GPU architecture can be checked before relying on the old upstream PyTorch build. Start an interactive allocation as in section 2, then run:

```bash
cd "$WORKFLOW_DIR"
bash scripts/setup_env.sh
```

The helper puts package caches/environment records in scratch, creates a prefix environment from [`environment.yml`](environment.yml), activates it, import-checks the training/inference runtime (including the Lightning CLI, progress bar, and TensorBoard logger), prints package/CUDA status, and records `conda list --explicit` and `pip freeze` under `execution/`. Current pins include Python 3.8.16, PyTorch 1.12.1, torchvision 0.13.1, CUDA runtime 11.3.1, PyTorch Lightning 1.8.0, `jsonargparse` 4.19.0, Rich 13.3.1, TensorBoard 2.11.2, timm 0.6.12, torchdata 0.4.1, and the listed preprocessing libraries.

Expected check:

```text
Python 3.8.16
PyTorch: 1.12.1
CUDA runtime: 11.3.1
CUDA available: True
```

CUDA `True` is expected only inside an allocated GPU job. This old stack targets GPUs supported by its CUDA/PyTorch build; do not assume it works on Hopper/H100-or-newer. Check `nvidia-smi` and `torch.cuda.get_device_capability()` on the allocated node. If kernels are unsupported, stop and use a separately tested compatible stack rather than continuing with a CPU fallback or claiming GPU inference.

If `module load miniforge` fails, inspect the cluster's available modules (`module avail miniforge` or `module spider miniforge`) and follow the [current ICE software/module instructions](https://docs.pace.gatech.edu/ice/). If the module is unavailable, contact PACE support; do not guess a module name or install Miniforge over a shared cluster installation.

## 5. Download and verify the checkpoint

Set cache and artifact paths before downloading:

```bash
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export TORCH_HOME="$HOME/scratch/.cache/torch"
export GIT_CONFIG_GLOBAL="$HOME/scratch/.gitconfig"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$TORCH_HOME"

mkdir -p "$PROJECT_DIR/checkpoints"
export PRETRAINED_CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
curl --fail --location --retry 3 --retry-all-errors \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  --output "$PRETRAINED_CHECKPOINT"
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$PRETRAINED_CHECKPOINT" \
  | sha256sum --check -
stat --printf='checkpoint_bytes=%s\n' "$PRETRAINED_CHECKPOINT"
```

Expected: `5.625deg.ckpt: OK` and exactly `432419871` bytes. If the digest differs, remove only this incomplete/incorrect named file and download it again. Do not rename a different Hugging Face checkpoint to make the workflow continue.

## 6. Download and preprocess ERA5

### Obtain the raw files

1. Open the [TUM WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895) and follow its access/download instructions. The archive is an external dataset; this repository does not include the data.
2. Stage/extract the raw `5.625deg` ERA5 tree to scratch at `$PROJECT_DIR/data/5.625deg`. Ensure there is a `constants.nc` file and per-variable subdirectories with NetCDF files.
3. At a minimum the configured forecast requires `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, and pressure-level files for `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `temperature`, `relative_humidity`, and `specific_humidity`. The preprocessor also reads `constants.nc` for `land_sea_mask`, `orography`, and `lattitude`.
4. Preserve the archive's original filenames. Confirm that it contains all the years requested below. If you do not have access to the full 1979–2018 range, adjust split years explicitly and document the deviation.

Expected raw tree (variable names are directory names):

```text
$PROJECT_DIR/data/5.625deg/
  constants.nc
  2m_temperature/*.nc
  10m_u_component_of_wind/*.nc
  10m_v_component_of_wind/*.nc
  geopotential/*.nc
  u_component_of_wind/*.nc
  v_component_of_wind/*.nc
  temperature/*.nc
  relative_humidity/*.nc
  specific_humidity/*.nc
```

### Create the prepared data

The upstream preprocessing computes training means/stds and validation/test climatologies, converts NetCDF to hourly `.npz` shards, and saves latitude/longitude. Run on an ICE compute allocation with enough CPU, memory, and scratch; this is a CPU preprocessing job, not an inference command. Use your permitted CPU partition/account and request according to the input data size. For an interactive CPU allocation, set `CPU_PARTITION` to a partition confirmed by `sinfo`/PACE:

```bash
export PACE_ACCOUNT="<YOUR_PACE_ACCOUNT>"
export CPU_PARTITION="<YOUR_CPU_PARTITION>"
srun --account="$PACE_ACCOUNT" --partition="$CPU_PARTITION" \
  --cpus-per-task=16 --mem=128G --time=12:00:00 --pty bash
```

In that shell:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export ERA5_RAW="$PROJECT_DIR/data/5.625deg"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"

python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$ERA5_RAW" \
  --save_dir "$ERA5_NPZ" \
  --start_train_year 1979 \
  --start_val_year 2016 \
  --start_test_year 2017 \
  --end_year 2019 \
  --num_shards 8 \
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

Here `end_year` is exclusive: train is 1979–2015, validation is 2016, and test is 2017–2018. The upstream script assumes 365 days per year and truncates leap-year hours to 8760; its arrays are hourly. Eight shards per year gives 1095 hourly timesteps per shard.

The forecast uses 48 input variables in the exact order configured upstream: the three constant fields `land_sea_mask`, `orography`, `lattitude`; the surface fields `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`; and `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `temperature`, `relative_humidity`, and `specific_humidity` at each of 50, 250, 500, 600, 700, 850, and 925 hPa. It predicts five outputs: `geopotential_500`, `temperature_850`, `2m_temperature`, `10m_u_component_of_wind`, and `10m_v_component_of_wind`. Keep the spelling `lattitude` and order from the pinned config.

Expected prepared tree:

```text
$ERA5_NPZ/
  train/*.npz
  val/*.npz
  val/climatology.npz
  test/*.npz
  test/climatology.npz
  normalize_mean.npz
  normalize_std.npz
  lat.npy
  lon.npy
```

Check existence and an expected shard before starting GPU training:

```bash
test -s "$ERA5_NPZ/normalize_mean.npz"
test -s "$ERA5_NPZ/normalize_std.npz"
test -s "$ERA5_NPZ/lat.npy"
test -s "$ERA5_NPZ/lon.npy"
test -s "$ERA5_NPZ/test/climatology.npz"
find "$ERA5_NPZ/train" -maxdepth 1 -name '*.npz' -type f | head
find "$ERA5_NPZ/val" -maxdepth 1 -name '*.npz' -type f | head
find "$ERA5_NPZ/test" -maxdepth 1 -name '*.npz' -type f | head
du -sh "$ERA5_RAW" "$ERA5_NPZ"
df -h "$HOME/scratch"
```

If any `test -s` command fails or a split has no shard, do not submit training. Check the raw archive/year range and the preprocessing log first. Never use `test/climatology.npz` as an input shard.

## 7. Fine-tune the global-forecast task

After returning from the CPU preprocessing allocation to the login shell, re-export the paths needed by SLURM. Shell variables set inside an exited `srun` shell do not propagate back to the login shell. The SLURM log directory has to exist **before** job submission:

```bash
export WORKFLOW_REPO="$HOME/scratch/climax-inference"
export WORKFLOW_DIR="$WORKFLOW_REPO/workflows/climax-global-forecast"
export PROJECT_DIR="$HOME/scratch/climax-reproduction"
export ENV_PREFIX="$HOME/scratch/.conda/envs/climax-inference"
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
export PRETRAINED_CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"
export PACE_ACCOUNT="<YOUR_PACE_ACCOUNT>"
mkdir -p "$WORKFLOW_DIR/logs"
export WORKFLOW_DIR PROJECT_DIR ENV_PREFIX CLIMAX_SOURCE PRETRAINED_CHECKPOINT ERA5_NPZ
export TRAIN_DIR="$PROJECT_DIR/runs/global-forecast"
```

Submit with your PACE account:

```bash
cd "$WORKFLOW_DIR"
sbatch --account="$PACE_ACCOUNT" slurm/finetune.sbatch
squeue --me
```

The script loads Miniforge and activates the scratch environment in the batch shell; it does not depend on an interactive shell's `conda activate` carrying over. It runs the pinned upstream `src/climax/global_forecast/train.py` and config, initializes with the verified pretraining checkpoint, uses the upstream five output variables and 72-hour lead, and fine-tunes for up to 50 epochs. The upstream config has early stopping and writes Lightning logs/checkpoints beneath `$TRAIN_DIR`.

This full run is intentionally more substantial than a shape-only smoke test. Queue time and elapsed time depend on the GPU/data. If your account cannot run this allocation as requested, record the scheduler failure, make a smaller **explicit** pilot (fewer epochs/training batches), and label its scope in the evidence; do not describe a short pilot as the paper's full training experiment.

If the job reaches the wall-time limit after writing `checkpoints/last.ckpt`, it can be resumed by exporting the exact checkpoint and resubmitting:

```bash
export RESUME_CHECKPOINT="$TRAIN_DIR/checkpoints/last.ckpt"
sbatch --account="$PACE_ACCOUNT" slurm/finetune.sbatch
```

Record the original and resumed job IDs. The script passes this checkpoint to Lightning's trainer resume option and errors if the file is missing.

Monitor and retain scheduler evidence:

```bash
squeue --me
JOB_ID="<fine-tuning-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -n 80 "$WORKFLOW_DIR/logs/climax-finetune-$JOB_ID.out"
```

The trainer automatically evaluates the best validation checkpoint at the end. Inspect the job log for the best-checkpoint restore message, and list candidates:

```bash
JOB_ID="<fine-tuning-job-id>"
find "$TRAIN_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -print
grep -E 'Restoring states from best checkpoint path|best_model_path|test/' \
  "$WORKFLOW_DIR/logs/climax-finetune-$JOB_ID.out"
```

Set `FINETUNED_CHECKPOINT` to the **exact saved checkpoint selected by validation** (typically `epoch_*.ckpt`, not the initial `5.625deg.ckpt`). Retain the job log so the selection is auditable.

## 8. Run held-out inference

Set the selected model and output location, then submit the inference job:

```bash
export FINETUNED_CHECKPOINT="/absolute/path/to/the/validation-selected/epoch_checkpoint.ckpt"
export OUTPUT_DIR="$PROJECT_DIR/runs/inference-test-2017"
cd "$WORKFLOW_DIR"
sbatch --account="$PACE_ACCOUNT" slurm/infer.sbatch
squeue --me
```

The batch script uses the `test` split, the first sorted test shard, 72-hour lead, and 32 inputs separated by 24 hourly indices (approximately one forecast sample per day within that 45-day shard). It loads the fine-tuned Lightning checkpoint strictly: missing/unexpected state-dict keys are errors. The script also verifies the initial pretraining checkpoint SHA-256 to record provenance.

For an interactive GPU allocation, the equivalent command is:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export CLIMAX_COMMIT="$(git -C "$CLIMAX_SOURCE" rev-parse HEAD)"
python "$WORKFLOW_DIR/scripts/infer.py" \
  --climax-source "$CLIMAX_SOURCE" \
  --config "$CLIMAX_SOURCE/configs/global_forecast_climax.yaml" \
  --checkpoint "$FINETUNED_CHECKPOINT" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --data-dir "$ERA5_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --split test --time-index 0 --lead-hours 72 \
  --num-samples 32 --sample-stride 24 --device cuda
```

The run records both code commits and the output hash in `metrics.json`. The run should produce:

- `climax-forecast.npz`: predictions and observed targets with shape `[32, 5, 32, 64]`, persistence and climatology baselines, variable names, coordinates, lead, and sample indices.
- `metrics.json`: smoke-test outcome, per-variable RMSE for ClimaX/persistence/climatology, relative improvement against persistence, prediction/target ranges, checkpoint/config/output digests, source/workflow commits, GPU, training checkpoint epoch/step, and forward-pass runtime.
- `logs/climax-inference-<JOB_ID>.out` and `.err`: scheduler output and errors.

## 9. Verify the result and interpret the margin

First check the completed job state, outputs, metadata, dimensions, and finite values:

```bash
JOB_ID="<inference-job-id>"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat "$OUTPUT_DIR/metrics.json"
ls -lh "$OUTPUT_DIR"
```

The JSON smoke test must report `status: "passed"`, exact output shape `[32, 5, 32, 64]`, and `true` for finite predictions and targets. The fine-tuned checkpoint must load strictly; the expected upstream initialization checkpoint hash must match; and the saved test shard, time indices, and 72-hour lead must be consistent. These checks catch broken loads, wrong output dimensions, NaNs/infinities, and invalid sample windows.

The substantive comparison is the per-variable **latitude-weighted RMSE** against the same ERA5 target. The script uses the upstream metric's convention: calculate a cosine-latitude-weighted spatial RMSE for each sample and average those sample RMSE values:

- `climax_weighted_rmse`: fine-tuned ClimaX result.
- `persistence_weighted_rmse`: use the input field as the forecast.
- `climatology_weighted_rmse`: test-split mean field produced by the upstream preprocessor.
- `climax_vs_persistence_relative_improvement`: positive means ClimaX RMSE is lower than persistence; negative means persistence is better. This is a fraction (multiply by 100 for percent).

For interpretation, compare the model with both baselines **for every output variable**, not only an aggregate. The expected qualitative result of a successful fine-tune is lower aggregate error than simple persistence on most variables, with plausible denormalized field ranges. This is not guaranteed for a failed/short training run. A negative improvement is a reported result to investigate, not a number to hide. Climatology uses the mean of the test split and is descriptive, not an independent training-free forecast baseline.

There is no official absolute RMSE tolerance for this exact PACE run in the pinned upstream guide. A single universal numeric cutoff would be misleading because fine-tuning data volume, checkpoint/epoch selection, ERA5 versions, and evaluation windows affect scores. Use the upstream trainer's held-out `test/w_rmse` plus the per-variable baseline table as the expected-margin evidence. Report observed scores, relative differences, and any validation/test gap; only claim a model advantage when the measured scores support it.

As a broad plausibility screen only (not a hard metric threshold), outputs should be finite and in recognizable physical units after denormalization: geopotential at 500 hPa is typically on the order of tens of thousands of m²/s², temperatures are in kelvin (roughly a few hundred), and wind components are in m/s with both signs. If values are orders of magnitude away, suspect normalization/denormalization or checkpoint-loading errors. Do not treat these broad ranges as a substitute for the ERA5 metric comparison.

The inference script emits summary ranges/metrics, not a plotted map. For visual inspection, load the NPZ on a local machine with NumPy/Matplotlib and plot a prediction, its matching target, and their difference for the same sample/variable; include the figure only if generated from this actual run.

## 10. Record execution evidence

No PACE run has yet been executed from this checkout, so there are no measured GPU model, wall time, memory, RMSE, output file, or job log values to report here. After your run, fill in [`execution/README.md`](execution/README.md) with the dated run record and commit/link the relevant non-large, non-sensitive evidence: job stdout/stderr, explicit environment export, source commit, checkpoint hash, preprocessing details, metrics JSON, and a small output inventory. Do not commit the ERA5 dataset or either checkpoint.

Useful job evidence commands:

```bash
JOB_ID="<job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
nvidia-smi
sha256sum "$PRETRAINED_CHECKPOINT" "$FINETUNED_CHECKPOINT"
```

Capture `nvidia-smi` while the job is allocated; after the job ends, it may not report the job's GPU. Keep failed logs too and explain what failed and how you responded.

## 11. Troubleshooting

### `-bash: conda: command not found`

Run `module load miniforge`, then `source "$(conda info --base)/etc/profile.d/conda.sh"` in the current shell. Run `command -v conda`. For batch jobs, the checked-in scripts perform those steps explicitly. If the module name is not available, consult PACE's current module guide/support rather than copying an unverified module name.

### Conda solve/install failure or `No space left on device`

Check that you loaded Miniforge, scratch is mounted/writable, and cache variables point to scratch. Inspect the full solver output; do not mix unpinned PyTorch/torchvision versions. Preserve the error and any generated package exports. If creation stopped partway, inspect the exact `$ENV_PREFIX`; only after saving relevant logs, remove that specific incomplete environment with `conda env remove --prefix "$ENV_PREFIX"`, then rerun the setup helper. If needed, use PACE-supported Mamba/Conda instructions and record a separate, fully pinned environment rather than silently altering the checked-in one.

### `torch.cuda.is_available()` is false / CUDA kernel or architecture error

Check `squeue`/`sacct`, `nvidia-smi`, loaded GPU allocation, driver, `torch.version.cuda`, and GPU capability inside the compute job. A login node has no allocated GPU. PyTorch 1.12.1/CUDA 11.3.1 is the upstream-era pin and may not support newer GPU architectures. Stop if the first CUDA operation fails; do not accept CPU fallback as a GPU run. Ask PACE which GPU/driver stack is available and validate any newer PyTorch/CUDA version against ClimaX before documenting it.

### Checkpoint download or strict load fails

For the initial checkpoint, verify exact size and pinned SHA-256. For the fine-tuned checkpoint, ensure `FINETUNED_CHECKPOINT` is the validation-selected Lightning `.ckpt` emitted by this pinned source run, not the `5.625deg.ckpt` initialization file. Preserve the full load error; never use `strict=False` to hide incompatibilities.

### Missing input variables or shape mismatch

Check that all raw variable directories and requested years were present during preprocessing; inspect the conversion log. Ensure normalized ERA5 data is 5.625°, the correct `test` split is passed, and `normalize_mean.npz`, `normalize_std.npz`, `test/climatology.npz`, `lat.npy`, and `lon.npy` exist. Keep the variable spelling and order from the upstream config, especially `lattitude`.

### Fine-tuning is out of memory or exceeds wall time

Record the allocated GPU model/memory, job ID, epoch/step, and failure log. Retry with a lower `--data.batch_size` (the committed job starts at 4), smaller documented pilot settings, or a larger-memory allocation permitted by ICE. Any reduced-data/epoch run must be labeled a pilot and cannot be directly compared to a full upstream run. Do not alter precision, data splits, or checkpoint silently.

### Metrics are unexpectedly poor or worse than baselines

First confirm the fine-tuned checkpoint was selected by validation; the initial pretraining checkpoint alone is not an adequate forecast checkpoint. Check lead-hours (72), variable order, matching normalization files, raw ERA5 units, grid resolution, training log for nonzero loss/validation, and which test split/shard was evaluated. Report all results, including baseline wins, and compare with the full upstream trainer test metrics before drawing conclusions.

## 12. Cleanup

Retain the source checkout, environment, raw/preprocessed data, checkpoints, and run outputs until the evidence is reviewed and backed up in an approved location. Then remove only artifacts you no longer need:

```bash
# Inspect exact resolved paths before removing anything.
printf 'Run output: %s\n' "$OUTPUT_DIR"
printf 'Training output: %s\n' "$TRAIN_DIR"

# Optional, targeted cleanup after evidence is safely retained:
# rm -rf -- "$OUTPUT_DIR"
# rm -rf -- "$TRAIN_DIR"
# rm -f -- "$PRETRAINED_CHECKPOINT"
# conda env remove --prefix "$ENV_PREFIX"
```

Do not delete `$HOME/scratch`, the workflow checkout, or source ERA5 data recursively. Observe PACE project/scratch cleanup rules.

## Supporting documentation

- [PACE ICE login, SLURM, storage, and modules](https://docs.pace.gatech.edu/ice/)
- [ClimaX original usage guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [ClimaX installation guide](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/install.md)
- [ClimaX global forecasting config](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml)
- [ClimaX ERA5 preprocessing source](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/data_preprocessing/nc2np_equally_era5.py)
- [ClimaX global forecast trainer](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/global_forecast/train.py)
- [WeatherBench data archive](https://dataserv.ub.tum.de/index.php/s/m1524895)
- [Pinned ClimaX checkpoint](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt)

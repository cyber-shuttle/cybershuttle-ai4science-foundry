# Frozen ClimaX inference on Georgia Tech PACE ICE

This guide takes a student from an ICE login through environment setup, reduced ERA5 data preparation, frozen pretrained encoder inference, output verification, and cleanup. The primary task is representation extraction; it does not fine-tune ClimaX or make weather forecasts. The original full-data forecast experiment is retained later in this guide as an optional, separate workflow. See the [top-level reproduction guide](../../README.md) for the short path.

## What this run reproduces

The primary workflow **uses ClimaX's pretrained encoder directly** to extract representations from the single `geopotential_500` input variable:

1. Load the public 5.625° checkpoint strictly into the pinned upstream ClimaX pretraining module.
2. Encode 32 daily-spaced ERA5 geopotential samples on the 5.625° grid.
3. Save 512 spatial tokens of 1024 dimensions per sample, plus pooled 1024-dimensional vectors.
4. Validate finite/nonconstant output and deterministic repeat inference; retain hashes and PACE execution evidence.

This uses a reduced year range (train 2013–2015, validation 2016, test 2017) and a single ERA5 pressure-level variable group plus constants. These embeddings are not forecast predictions or a claim of downstream forecast skill.

## 1. Provenance and fixed artifacts

| Artifact | Source / identifier | Use |
|---|---|---|
| Model implementation | [Microsoft/ClimaX commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c) | Pretrained encoder and upstream preprocessing |
| Pretraining config | Pinned [`pretrain_climax.yaml`](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/pretrain_climax.yaml) | Encoder architecture, variable map, grid |
| Weights | [`tungnd/climax`, revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, `5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt) | Direct pretrained inference |
| Checkpoint SHA-256 | `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43` | Verify downloaded weights |
| Weather input | [WeatherBench ERA5 archive](https://dataserv.ub.tum.de/index.php/s/m1524895) | `geopotential_500`, split data and normalization |

The checkpoint is pretrained on CMIP6 and is directly used here as a frozen feature encoder. ClimaX's separate global-forecast task requires fine-tuning and is documented as an optional extension below.

## 2. PACE access and storage

You need a Georgia Tech PACE ICE account, permission to use a GPU allocation, an allocation/account name, and scratch storage for the environment, ClimaX source, the reduced ERA5 input, checkpoint, and output. Do not assume a particular project/account name. Replace `<YOUR_PACE_ACCOUNT>` below with an account assigned to you by PACE; `coc` may be appropriate for some users but is not universal.

Connect using the login host and authentication steps in the [PACE ICE documentation](https://docs.pace.gatech.edu/ice/). Check which allocation/account is assigned to you, then use a GPU allocation such as:

```bash
ssh "<your_GT_username>@login-ice.pace.gatech.edu"
sacctmgr show associations user="$USER"
srun --account="<YOUR_PACE_ACCOUNT>" --partition=ice-gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

Replace placeholders with your PACE username and assigned account; confirm `ice-gpu` and the resource limits with current PACE guidance. The primary embedding job requests one GPU, 4 CPUs, 32 GiB RAM, and 1 hour. CPU preprocessing has a separate suggested request below. Record actual resources, GPU model, elapsed time, and peak memory in the execution record.

Use PACE scratch for caches and large files. The commands use `$HOME/scratch`; confirm its availability/quota for your account. Conda/pip/temp/model caches and datasets/checkpoints are kept there rather than in the small home quota.

## 3. Clone this workflow and the pinned model source

Run on ICE. Clone the published `climax-inference` assignment branch:

```bash
git clone --branch climax-inference \
  https://github.com/purvikathalkar/climax-inference.git \
  "$HOME/scratch/climax-inference"
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$HOME/scratch/climax-reproduction"
git clone https://github.com/microsoft/ClimaX.git \
  "$HOME/scratch/climax-reproduction/ClimaX"
git -C "$HOME/scratch/climax-reproduction/ClimaX" \
  checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
git -C "$HOME/scratch/climax-reproduction/ClimaX" rev-parse HEAD
```

If `git clone --branch climax-inference` says the branch does not exist, the branch has not yet been pushed to GitHub. Use the repository's published assignment branch once available; do not silently clone `main` and claim it is the reproduction revision.

## 4. Initialize Conda and create the environment

On PACE, Conda is supplied through the Miniforge environment module. A new login shell does not necessarily have a `conda` command on `PATH`; this is why running `conda env create` immediately can return `-bash: conda: command not found`.

In each fresh shell that uses Conda, initialize Miniforge once:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda --version
```

Configure named Conda environments to live in scratch once:

```bash
mkdir -p "$HOME/scratch/.conda/envs"
conda config --add envs_dirs "$HOME/scratch/.conda/envs"
conda config --show envs_dirs
```

Create the environment **inside a GPU allocation**, so CUDA visibility and GPU compatibility are checked before embedding inference. Start an allocation as in section 2, then run:

```bash
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
bash scripts/setup_env.sh
conda activate climax-inference
```

The setup helper prefers Mamba for faster dependency solving, falling back to Conda with an explicit slower-solver notice if Mamba is not available from the loaded Miniforge module. It creates/updates the environment at `$HOME/scratch/.conda/envs/climax-inference` and redirects Conda, pip, temporary, Hugging Face, and PyTorch caches to scratch. It pins pip 24.0 and MKL 2024.0, checks the Lightning CLI/logger imports, records explicit Conda and pip package lists in `execution/`, and runs a real CUDA tensor operation.

There is no separate ClimaX CUDA-extension build step in this workflow. The pinned model runs with the prebuilt PyTorch/CUDA packages; do not apply unrelated GCC symlink, TensorRT, CUTLASS, or custom-extension instructions from other model guides.

### Environment pins and updating an existing environment

Important pins are Python 3.8.16, PyTorch 1.12.1, torchvision 0.13.1, CUDA runtime 11.3.1, pip 24.0, MKL 2024.0, PyTorch Lightning and lightning-lite 1.8.0, `jsonargparse` 4.19.0, Rich 13.3.1, TensorBoard 2.11.2, timm 0.6.12, and torchdata 0.4.1. The complete package specification is in [`environment.yml`](environment.yml).

For reference, the setup helper's Mamba operations are equivalent to:

```bash
mamba env create \
  -p "$HOME/scratch/.conda/envs/climax-inference" \
  -f "$HOME/scratch/climax-inference/workflows/climax-global-forecast/environment.yml"

"$HOME/scratch/.conda/envs/climax-inference/bin/python" \
  -m pip install "pip==24.0"

mamba env update \
  -p "$HOME/scratch/.conda/envs/climax-inference" \
  -f "$HOME/scratch/climax-inference/workflows/climax-global-forecast/environment.yml"

mamba install \
  -p "$HOME/scratch/.conda/envs/climax-inference" \
  -c conda-forge "mkl=2024.0"
```

Run `mamba env create` only the first time. When dependencies in `environment.yml` change or a previous solve left the environment incomplete, run `mamba env update` and then reapply the explicit pip/MKL pins shown above. The setup helper performs the required operations automatically. If using the `mamba install` command manually, use the exact version `mkl=2024.0`; do not add a `.*` suffix, which triggers Conda's deprecated wildcard-version warning.

Verify the versions and CUDA runtime/device from the allocated GPU shell:

```bash
conda activate climax-inference
python --version
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

Expected check:

```text
Python 3.8.16
PyTorch: 1.12.1
CUDA runtime: 11.3.1
CUDA available: True
```

CUDA `True` is expected only inside an allocated GPU job. This old stack may not support newer GPU architectures. Check `nvidia-smi` and the reported CUDA capability on the allocated node. If the CUDA tensor test or imports fail, stop; do not silently continue with CPU fallback or assume that newer PyTorch versions are drop-in replacements.

If `module load miniforge` fails, inspect the cluster's available modules (`module avail miniforge` or `module spider miniforge`) and follow the [current ICE software/module instructions](https://docs.pace.gatech.edu/ice/). If it remains unavailable, contact PACE support; do not install a second Conda distribution over the shared cluster setup.

## 5. Download and verify the checkpoint

Download the pinned initialization checkpoint from the project directory:

```bash
mkdir -p "$HOME/scratch/climax-reproduction/checkpoints"
curl --fail --location --retry 3 --retry-all-errors \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  --output "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt"
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt" \
  | sha256sum --check -
stat --printf='checkpoint_bytes=%s\n' "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt"
```

Expected: `5.625deg.ckpt: OK` and exactly `432419871` bytes. If the digest differs, remove only this incomplete/incorrect named file and download it again. Do not rename a different Hugging Face checkpoint to make the workflow continue.

## 6. Download and preprocess the reduced ERA5 input

### Obtain the raw files

The primary inference run needs only the WeatherBench `5.625deg/geopotential` pressure-level group and `constants.nc`, rather than every ERA5 variable. This group contains `geopotential_500`; ClimaX's pinned preprocessing script expects the parent pressure-level group and extracts that level. The exact download size depends on TUM's current archive options; verify the selected path before starting a long transfer.

The WeatherBench README also offers a single-level `geopotential_500` ZIP, but that ZIP is not the input layout expected by the pinned ClimaX preprocessor. Do not rename its directory or assume it is directly compatible. The parent `geopotential` group is the supported reduced-data route.

Stage the archive files with their original names:

```text
$HOME/scratch/climax-reproduction/data/5.625deg/
  constants.nc
  geopotential/*.nc
```

The reduced pilot uses only 2013–2017. If the archive exposes yearly files, download those years only. If it provides one ZIP for all years of this variable group, it is still smaller than downloading all WeatherBench variables, though its size may remain substantial.

### Create the prepared data

The upstream preprocessing computes training means/stds and validation/test climatologies, converts NetCDF to hourly `.npz` shards, and saves latitude/longitude. Run on an ICE compute allocation with enough CPU, memory, and scratch; this is a CPU preprocessing job, not an inference command. Use your permitted CPU partition/account and request according to the input data size. For an interactive CPU allocation, set `CPU_PARTITION` to a partition confirmed by `sinfo`/PACE:

```bash
srun --account="<YOUR_PACE_ACCOUNT>" --partition="<YOUR_CPU_PARTITION>" \
  --cpus-per-task=16 --mem=128G --time=12:00:00 --pty bash
```

In that shell:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate climax-inference

python "$HOME/scratch/climax-reproduction/ClimaX/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$HOME/scratch/climax-reproduction/data/5.625deg" \
  --save_dir "$HOME/scratch/climax-reproduction/data/geopotential_npz" \
  --start_train_year 2013 \
  --start_val_year 2016 \
  --start_test_year 2017 \
  --end_year 2018 \
  --num_shards 8 \
  --variables geopotential
```

Here `end_year` is exclusive: train is 2013–2015, validation is 2016, and test is 2017. The upstream script assumes 365 days per year and truncates leap-year hours to 8760; its arrays are hourly. Eight shards per year gives 1095 hourly timesteps per shard. Report this reduced period as a pilot, not as a full-data benchmark.

The encoder receives one normalized channel: `geopotential_500`. Constants are prepared by the upstream converter but are not passed into the encoder in this example. The pinned pretraining model supports this variable identifier.

Expected prepared tree:

```text
$HOME/scratch/climax-reproduction/data/geopotential_npz/
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

Check existence and an expected shard before starting GPU inference:

```bash
ERA5_NPZ="$HOME/scratch/climax-reproduction/data/geopotential_npz"
test -s "$ERA5_NPZ/normalize_mean.npz"
test -s "$ERA5_NPZ/normalize_std.npz"
test -s "$ERA5_NPZ/lat.npy"
test -s "$ERA5_NPZ/lon.npy"
test -s "$ERA5_NPZ/test/climatology.npz"
find "$ERA5_NPZ/train" -maxdepth 1 -name '*.npz' -type f | head
find "$ERA5_NPZ/val" -maxdepth 1 -name '*.npz' -type f | head
find "$ERA5_NPZ/test" -maxdepth 1 -name '*.npz' -type f | head
du -sh "$HOME/scratch/climax-reproduction/data/5.625deg" "$ERA5_NPZ"
df -h "$HOME/scratch"
```

If any `test -s` command fails or a split has no shard, do not submit inference. Check the raw archive/year range and preprocessing log first. Never use `test/climatology.npz` as an input shard.

## 7. Run frozen pretrained embedding inference

Return to the login shell, load Miniforge, and submit the inference job:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

The job requests one GPU, 4 CPUs, 32 GiB RAM, and one hour. It strictly loads the pinned pretrained checkpoint into ClimaX's upstream `PretrainModule`; no fine-tuned forecast checkpoint or training stage is involved. The script encodes 32 samples from the first sorted test shard of `geopotential_500`. Outputs are written to `$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>/`.

For an interactive GPU allocation, run the same command directly:

```bash
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/embed.py" \
  --climax-source "$HOME/scratch/climax-reproduction/ClimaX" \
  --config "$HOME/scratch/climax-reproduction/ClimaX/configs/pretrain_climax.yaml" \
  --checkpoint "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt" \
  --data-dir "$HOME/scratch/climax-reproduction/data/geopotential_npz" \
  --output-dir "$HOME/scratch/climax-reproduction/runs/manual-embeddings" \
  --variable geopotential_500 --split test --time-index 0 \
  --num-samples 32 --sample-stride 24 --device cuda
```

The lead-time embedding is set to zero because this task extracts representations rather than forecast outputs. `climax-embeddings.npz` contains the spatial tokens `[32, 512, 1024]` and pooled vectors `[32, 1024]`. `metrics.json` records the smoke check, repeat-inference determinism, representation diagnostics, checkpoint/config/data provenance, and output hash. Finite/nonconstant features and deterministic repeat inference demonstrate a valid technical execution, not forecast skill.

Check the job state and evidence:

```bash
JOB_ID="<embedding-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
cat "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID/metrics.json"
ls -lh "$HOME/scratch/climax-reproduction/runs/embeddings-$JOB_ID"
```

## Optional forecast extension (separate fine-tuning experiment)

The remaining forecast procedure is **not required** for the primary frozen-encoder inference workflow. It predicts future ERA5 fields and requires the full forecast input set plus a fine-tuned checkpoint. Do not use this path if the goal is simply to extract pretrained representations.

For this separate experiment, the primary reduced `geopotential_npz` dataset is insufficient. Retrieve the full raw input set described by the pinned [`global_forecast_climax.yaml`](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml): `constants.nc`, surface groups `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, and pressure-level groups `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `temperature`, `relative_humidity`, and `specific_humidity`. Preprocess those full inputs into a separate `$HOME/scratch/climax-reproduction/data/5.625deg_npz` directory using train 1979–2015, validation 2016, test 2017–2018 (`--end_year 2019`) and all nine `--variables` flags. Keep the small embedding data separate from this optional forecast dataset.

### Fine-tune the global-forecast task

After returning from the CPU allocation, return to the login shell. Shell state from an exited `srun` shell is not retained, so initialize Miniforge again. The batch scripts use the standard scratch paths shown in this guide and do not require user-defined path exports. Create the SLURM log directory **before** submission because SLURM opens its log files before the script starts:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/finetune.sbatch
squeue --me
```

The script loads Miniforge and activates `climax-inference` in the batch shell; it does not depend on an interactive `conda activate` carrying over. It runs the pinned upstream trainer/config, initializes with the verified pretraining checkpoint, uses the upstream five output variables and 72-hour lead, and fine-tunes for up to 50 epochs. The upstream config has early stopping and writes Lightning logs/checkpoints beneath `$HOME/scratch/climax-reproduction/runs/global-forecast`.

This full run is intentionally more substantial than a shape-only smoke test. Queue time and elapsed time depend on the GPU/data. If your account cannot run this allocation as requested, record the scheduler failure, make a smaller **explicit** pilot (fewer epochs/training batches), and label its scope in the evidence; do not describe a short pilot as the paper's full training experiment.

If the job reaches the wall-time limit after writing `checkpoints/last.ckpt`, resume from that checkpoint by passing its path as the optional job argument:

```bash
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/finetune.sbatch \
  "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints/last.ckpt"
```

Record the original and resumed job IDs. The script passes this checkpoint to Lightning's trainer resume option and errors if the file is missing.

Monitor and retain scheduler evidence:

```bash
squeue --me
JOB_ID="<fine-tuning-job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
tail -n 80 "$HOME/scratch/climax-inference/workflows/climax-global-forecast/logs/climax-finetune-$JOB_ID.out"
```

The trainer automatically evaluates the best validation checkpoint at the end. Inspect the job log for the best-checkpoint restore message, and list candidates:

```bash
JOB_ID="<fine-tuning-job-id>"
find "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints" \
  -maxdepth 1 -type f -name '*.ckpt' -print
grep -E 'Restoring states from best checkpoint path|best_model_path|test/' \
  "$HOME/scratch/climax-inference/workflows/climax-global-forecast/logs/climax-finetune-$JOB_ID.out"
```

Use the **exact saved checkpoint selected by validation** (typically `epoch_*.ckpt`, not the initial `5.625deg.ckpt`). Retain the job log so the selection is auditable.

## 8. Run held-out inference

Pass the selected model checkpoint as the inference job's argument:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/infer.sbatch \
  "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints/epoch_XXX.ckpt"
squeue --me
```

Replace `epoch_XXX.ckpt` with the validation-selected filename. The batch script uses the `test` split, the first sorted test shard, 72-hour lead, and 32 inputs separated by 24 hourly indices (approximately one forecast sample per day within that 45-day shard). Results go to `$HOME/scratch/climax-reproduction/runs/inference-<job-id>`. It loads the fine-tuned Lightning checkpoint strictly: missing/unexpected state-dict keys are errors. The script also verifies the initial pretraining checkpoint SHA-256 to record provenance.

For an interactive GPU allocation, the equivalent command is:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate climax-inference
python "$HOME/scratch/climax-inference/workflows/climax-global-forecast/scripts/infer.py" \
  --climax-source "$HOME/scratch/climax-reproduction/ClimaX" \
  --config "$HOME/scratch/climax-reproduction/ClimaX/configs/global_forecast_climax.yaml" \
  --checkpoint "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints/epoch_XXX.ckpt" \
  --pretrained-checkpoint "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt" \
  --data-dir "$HOME/scratch/climax-reproduction/data/5.625deg_npz" \
  --output-dir "$HOME/scratch/climax-reproduction/runs/manual-inference" \
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
cat "$HOME/scratch/climax-reproduction/runs/inference-$JOB_ID/metrics.json"
ls -lh "$HOME/scratch/climax-reproduction/runs/inference-$JOB_ID"
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

## 10. Execution evidence and run record

Use the following record to capture each run. Preserve job stdout/stderr, Conda package exports, source commit, checkpoint hashes, preprocessing details, metrics JSON, and a small output inventory. Do not commit ERA5 data or model checkpoints.

### PACE setup/allocation evidence supplied so far

On 2026-10-11, PACE reported the `coc` account for the `ice-slurm` cluster with the `coc-ice` QOS. Interactive allocation job `6196618` was queued and allocated on `ice-gpu` with one GPU, 8 CPUs, 32 GiB RAM, and a two-hour limit. The SLURM prolog reported a start time of 2026-10-11 21:05:44. This is allocation/setup evidence; add the embedding job's logs, output, and metrics from the actual run record when available.

### Run record template

Copy this section for each completed, partial, or failed training/inference run:

#### Run `<date>` / `<purpose>`

- Result: `success / failed / partial`
- PACE account, partition, QOS:
- Primary embedding SLURM job ID / state:
- Optional forecast fine-tuning/inference job IDs / states (if run):
- Compute node and GPU model / memory:
- Requested GPUs, CPUs, host memory, wall time:
- Observed elapsed time / peak RSS / peak GPU memory:
- ClimaX Git commit:
- Workflow Git commit:
- Pretraining checkpoint source revision and SHA-256:
- Fine-tuned checkpoint path, validation-selection rationale, SHA-256:
- ERA5 source/access date, years/splits, preprocessing command:
- Python, PyTorch, CUDA, Lightning, timm, pip, MKL versions:
- Training epochs completed / best validation `w_rmse`:
- Trainer held-out test metrics:
- Embedding smoke test status / spatial shape / pooled shape:
- Embedding finite, nonconstant, and deterministic-repeat checks:
- Embedding diagnostics (mean token standard deviation / pooled-vector norm):
- Forecast inference smoke test and output shape (optional experiment only):
- Per-variable forecast RMSE (optional forecast experiment only):
  - `geopotential_500`:
  - `temperature_850`:
  - `2m_temperature`:
  - `10m_u_component_of_wind`:
  - `10m_v_component_of_wind`:
- Output file inventory:
- Failures, troubleshooting, deviations, and follow-up:

Explain any metric worse than its baseline; retain unsuccessful/partial job logs rather than omitting them.

Useful job evidence commands:

```bash
JOB_ID="<job-id>"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
nvidia-smi
sha256sum \
  "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt" \
  "$HOME/scratch/climax-reproduction/runs/global-forecast/checkpoints/epoch_XXX.ckpt"
```

Capture `nvidia-smi` while the job is allocated; after the job ends, it may not report the job's GPU. Keep failed logs too and explain what failed and how you responded. For the primary run, preserve `climax-embeddings.npz` and its `metrics.json`; do not describe the representation diagnostics as a forecast score.

## 11. Troubleshooting

### `-bash: conda: command not found`

Run `module purge`, `module load miniforge`, and then `source "$(conda info --base)/etc/profile.d/conda.sh"` in the current shell. Check `command -v conda`. For batch jobs, the checked-in scripts perform those steps explicitly. If the module name is not available, consult PACE's current module guide/support rather than copying an unverified module name.

### Assignment branch or artifact download is unavailable

If `git clone --branch climax-inference` fails, check with the assignment owner for the published branch URL; do not proceed from a different branch without recording the revision. If network downloads fail on a compute node, use a PACE-approved transfer/login route and keep the downloaded files under scratch. For a checkpoint SHA mismatch, treat the file as incomplete or incorrect: remove only the named checkpoint file, download it again, and require the documented SHA-256 to pass before inference.

### Fresh shell cannot find an activated environment

Conda activation is shell-local; a fresh login or SLURM batch shell must load Miniforge and source `conda.sh` before `conda activate climax-inference`. The job scripts do this themselves. If Conda says the environment name is unknown, check `conda config --show envs_dirs` and ensure `$HOME/scratch/.conda/envs` is listed before recreating the environment.

### Conda solve/install failure or `No space left on device`

Check that Miniforge is loaded and scratch is writable. Inspect the full solver output; do not mix unpinned PyTorch/torchvision versions. Preserve the error and generated package exports. If creation stopped partway, inspect `conda env list`; after saving relevant logs, remove only the incomplete environment with `mamba env remove -p "$HOME/scratch/.conda/envs/climax-inference"` (or `conda env remove -p ...`), then rerun the setup helper. If needed, use PACE-supported Mamba/Conda instructions and record a separate, fully pinned environment rather than silently altering the checked-in one.

### Deprecated wildcard-version warning

If Conda/Mamba reports that using `.*` with a relational version operator is superfluous/deprecated, inspect the version expressions in the environment file or command. Use a concrete pin such as `mkl=2024.0` rather than `mkl=2024.0.*`. Keep a copy of the solver output when documenting dependency troubleshooting.

### `torch.cuda.is_available()` is false / CUDA kernel or architecture error

Check `squeue`/`sacct`, `nvidia-smi`, loaded GPU allocation, driver, `torch.version.cuda`, and GPU capability inside the compute job. A login node has no allocated GPU. PyTorch 1.12.1/CUDA 11.3.1 is the upstream-era pin and may not support newer GPU architectures. Stop if the first CUDA operation fails; do not accept CPU fallback as a GPU run. Ask PACE which GPU/driver stack is available and validate any newer PyTorch/CUDA version against ClimaX before documenting it.

### Checkpoint download or strict load fails

For the initial checkpoint, verify exact size and pinned SHA-256. For embedding inference, use the pinned `5.625deg.ckpt` and `configs/pretrain_climax.yaml`; the script loads it strictly into `PretrainModule`. For the separate optional forecast job, pass the validation-selected Lightning `.ckpt`, not the pretrained `5.625deg.ckpt`. Preserve the full load error; never use `strict=False` to hide incompatibilities.

### Missing input variables or shape mismatch

For the primary embedding task, check the `geopotential` directory, requested years, `geopotential_500` key in a test shard, and `normalize_mean.npz`, `normalize_std.npz`, `lat.npy`, and `lon.npy`. For the optional forecast experiment, ensure all configured surface/pressure-level groups exist and match the full global-forecast config. Keep variable identifiers exactly as listed in the pinned configs.

### Fine-tuning is out of memory or exceeds wall time

Record the allocated GPU model/memory, job ID, epoch/step, and failure log. Retry with a lower `--data.batch_size` (the committed job starts at 4), smaller documented pilot settings, or a larger-memory allocation permitted by ICE. Any reduced-data/epoch run must be labeled a pilot and cannot be directly compared to a full upstream run. Do not alter precision, data splits, or checkpoint silently.

### Metrics are unexpectedly poor or worse than baselines

This primary workflow does not calculate forecast metrics. If reviewing the optional forecast experiment, confirm that a validation-selected fine-tuned checkpoint was used; the pretraining checkpoint alone is not an adequate forecaster. Check lead time (72 hours), variable order, normalization files, raw ERA5 units, grid resolution, training logs, and test shard. Report baseline comparisons and trainer test metrics without hiding worse-than-baseline results.

## 12. Cleanup

Retain the source checkout, environment, raw/preprocessed data, checkpoints, and run outputs until the evidence is reviewed and backed up in an approved location. Then remove only artifacts you no longer need:

```bash
# Inspect the exact paths before removing anything.
printf '%s\n' "$HOME/scratch/climax-reproduction/runs/inference-<job-id>"
printf '%s\n' "$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>"
printf '%s\n' "$HOME/scratch/climax-reproduction/runs/global-forecast"

# Optional, targeted cleanup after evidence is safely retained:
# rm -rf -- "$HOME/scratch/climax-reproduction/runs/inference-<job-id>"
# rm -rf -- "$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>"
# rm -rf -- "$HOME/scratch/climax-reproduction/runs/global-forecast"
# rm -f -- "$HOME/scratch/climax-reproduction/checkpoints/5.625deg.ckpt"
# mamba env remove -p "$HOME/scratch/.conda/envs/climax-inference"
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

# ClimaX global forecast inference on PACE ICE

This workflow runs a single **6-hour, global forecast** using Microsoft's public ClimaX 5.625° pretrained checkpoint and one prepared ERA5 test shard. It is intended as a small, reproducible inference and validation run, not as a reproduction of the paper's full fine-tuning experiment.

The upstream repository documents the checkpoint as a pretrained ClimaX model and uses it to initialize a separately fine-tuned global-forecast model. This workflow applies the pretrained checkpoint directly to the forecast head and does **not** fine-tune it. Interpret the metrics accordingly; this run verifies that inference executes and compares predictions to the observed ERA5 target, persistence, and the test climatology. It does not claim the pretrained model is a validated operational weather forecaster.

## 1. Project overview and reproduction target

The run uses:

- **Model code:** Microsoft ClimaX, pinned to upstream commit [`6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`](https://github.com/microsoft/ClimaX/tree/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c).
- **Weights:** the public `5.625deg.ckpt` pretrained checkpoint from [`tungnd/climax`](https://huggingface.co/tungnd/climax), pinned to repository revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`.
- **Input:** a prepared ERA5 `test/*.npz` shard created by ClimaX's upstream preprocessing script.
- **Task:** predict the configured five output variables six hourly steps after an input time from the selected test shard.

The forecast config and preprocessing code live in the pinned ClimaX source checkout. This workflow doesn't include or commit the checkpoint or weather data.

## 2. Hardware requirements

The included SLURM template requests one GPU, four CPU cores, 32 GiB host memory, and two hours. It uses a single batch sample at ClimaX's 5.625° grid resolution (32 x 64). A GPU with at least 16 GiB VRAM is a reasonable starting point, but actual GPU memory and runtime must be recorded from the PACE ICE run; they have not been measured in this checkout.

Confirm the partition, account/project, GPU request syntax, and available GPU model with the ICE documentation or `sinfo` before submission. Add any required `#SBATCH --partition` / `--account` directives to [`slurm/infer.sbatch`](slurm/infer.sbatch). Do not assume that the generic `--gres=gpu:1` request matches every ICE partition.

## 3. Environment setup

Use the upstream-supported Python/PyTorch stack first. The versions are deliberately pinned to the ClimaX environment rather than upgraded speculatively; newer framework releases can change checkpoint loading and model behavior.

On an ICE login node with Conda or Mamba available:

```bash
git clone --branch climax-inference https://github.com/purvikathalkar/climax-inference.git
cd climax-inference/workflows/climax-global-forecast

conda env create -f environment.yml
conda activate climax-inference
```

The environment pins Python 3.8.16, PyTorch 1.12.1, torchvision 0.13.1, CUDA runtime 11.3.1, PyTorch Lightning 1.8.0, timm 0.6.12, torchdata 0.4.1, and the preprocessing/runtime packages in [`environment.yml`](environment.yml). The NVIDIA driver on the compute node must support the CUDA runtime. If the cluster's module stack conflicts with this environment or a package cannot be resolved, see [Troubleshooting](#8-troubleshooting) before changing versions.

Record the installed environment on the cluster for the evidence folder:

```bash
conda list --explicit > execution/conda-explicit.txt
python --version > execution/python-version.txt
python -c 'import torch; print("torch", torch.__version__); print("CUDA available", torch.cuda.is_available()); print("CUDA runtime", torch.version.cuda)' \
  > execution/torch-version.txt
```

## 4. Model and data retrieval

Set paths on a filesystem accessible to compute nodes. Put the ClimaX checkout, checkpoint, and preprocessed ERA5 data outside this Git repository because they are external/large artifacts.

```bash
export WORKFLOW="$PWD"
export PROJECT_DIR="$HOME/climax-reproduction"
mkdir -p "$PROJECT_DIR" "$WORKFLOW/execution" "$WORKFLOW/logs"

git clone https://github.com/microsoft/ClimaX.git "$PROJECT_DIR/ClimaX"
git -C "$PROJECT_DIR/ClimaX" checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c
git -C "$PROJECT_DIR/ClimaX" rev-parse HEAD | tee "$WORKFLOW/execution/climax-source-commit.txt"
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"

mkdir -p "$PROJECT_DIR/checkpoints"
export CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
curl -fL \
  'https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt' \
  -o "$CHECKPOINT"
echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  '"$CHECKPOINT" \
  | sha256sum -c - | tee "$WORKFLOW/execution/checkpoint-sha256.txt"
```

The SHA-256 is the Hugging Face LFS object hash at the pinned revision. The expected checkpoint size is 432,419,871 bytes.

Obtain the WeatherBench ERA5 data from the [TUM data archive linked by ClimaX](https://dataserv.ub.tum.de/index.php/s/m1524895). The original ClimaX guide and [ERA5 preprocessing script](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/data_preprocessing/nc2np_equally_era5.py) document the raw NetCDF directory structure, variables, split-year defaults, and conversion process. The raw WeatherBench data is large; check archive access/storage quota and stage it to project/scratch storage before conversion.

After downloading/extracting the `5.625deg` archive, preprocess it using the same pinned ClimaX checkout:

```bash
export ERA5_RAW="$PROJECT_DIR/data/5.625deg"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"

conda activate climax-inference
python "$CLIMAX_SOURCE/src/data_preprocessing/nc2np_equally_era5.py" \
  --root_dir "$ERA5_RAW" \
  --save_dir "$ERA5_NPZ" \
  --start_train_year 1979 \
  --start_val_year 2016 \
  --start_test_year 2017 \
  --end_year 2019 \
  --num_shards 8
```

The workflow uses `test/2017_0.npz` (or, if filenames differ, the first lexically sorted `.npz` under `test/`) and input index zero by default. The selected shard must contain all variables listed in the upstream [`global_forecast_climax.yaml`](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml), plus `normalize_mean.npz`, `normalize_std.npz`, `test/climatology.npz`, `lat.npy`, and `lon.npy`. The 6-hour target must be in the same shard. Preprocessing is computationally and storage intensive; these commands follow ClimaX defaults and may need appropriate CPU/memory allocations on ICE.

The model input variables are in the upstream config's exact order: `land_sea_mask`, `orography`, `lattitude`; `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`; and each of `geopotential`, `u_component_of_wind`, `v_component_of_wind`, `temperature`, `relative_humidity`, and `specific_humidity` at pressure levels 50, 250, 500, 600, 700, 850, and 925 hPa. Preserve the upstream spelling `lattitude` and ordering. The output variables are `geopotential_500`, `temperature_850`, `2m_temperature`, `10m_u_component_of_wind`, and `10m_v_component_of_wind`.

## 5. Inference

Set the paths exported above in the shell where you submit the job. The batch script expects:

```bash
export CLIMAX_SOURCE="$PROJECT_DIR/ClimaX"
export CHECKPOINT="$PROJECT_DIR/checkpoints/5.625deg.ckpt"
export ERA5_NPZ="$PROJECT_DIR/data/5.625deg_npz"
export OUTPUT_DIR="$WORKFLOW/results/run-001"
export CLIMAX_PYTHON="$(conda run -n climax-inference which python)"
```

Check that the test split contains a shard, then submit from the workflow directory:

```bash
cd "$WORKFLOW"
mkdir -p logs execution
sbatch slurm/infer.sbatch
squeue --me
```

For an interactive GPU allocation, run the same inference command directly:

```bash
"$CLIMAX_PYTHON" scripts/infer.py \
  --climax-source "$CLIMAX_SOURCE" \
  --config "$CLIMAX_SOURCE/configs/global_forecast_climax.yaml" \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$ERA5_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --time-index 0 \
  --lead-hours 6
```

`--split`, `--time-index`, and `--lead-hours` are selectable. The script validates the requested sample and variable arrays, normalizes inputs using the upstream statistics, runs one no-gradient forward pass, denormalizes predictions, and writes outputs. It records the checkpoint digest, code revision environment variable (if supplied), selected shard/index, runtime, device, and evaluation metrics.

## 6. Validation and output files

The run writes:

- `climax-forecast.npz`: predicted and observed fields, persistence and test-climatology baseline fields, variable names, latitude, longitude, and lead time.
- `metrics.json`: shape/finite checks and per-variable latitude-weighted RMSE for ClimaX, persistence, and test-climatology baselines, plus input/checkpoint/run metadata.
- `logs/climax-inference-<job-id>.out`: SLURM standard output and errors from the batch run.

The code treats successful tensor shape and finite-value checks as an inference smoke test. The substantive check is the comparison of each variable's latitude-weighted RMSE against the matching ERA5 target and two baselines. The climatology baseline is the upstream preprocessor's mean over the selected test split, so treat it as a descriptive reference rather than a strictly independent held-out baseline. These are measurements, not a promised pass threshold: record the observed metric values and explain whether the model is better or worse than each baseline. A finite result alone does not establish scientific forecast skill.

After the job completes:

```bash
cat "$OUTPUT_DIR/metrics.json"
ls -lh "$OUTPUT_DIR"
```

Run `nvidia-smi` during an interactive run or capture it in the batch job if site policy permits. Record the node/GPU type, SLURM job ID, start/end time, wall time, requested resources, observed peak GPU memory if available, and any errors in [`execution/README.md`](execution/README.md). Keep generated logs/results out of version control until reviewed; commit only appropriately sized, non-sensitive evidence and summaries.

## 7. Execution evidence status

This repository currently provides the workflow code and instructions only. **No PACE ICE inference job has been run from this checkout yet.** Do not represent example/expected output as execution evidence. Once an actual run completes, add a concise run record to [`execution/README.md`](execution/README.md) and include the relevant job log, metrics JSON, and output inventory (or a small representative result artifact if allowed by repository/data policy).

## 8. Troubleshooting

### A dependency fails to install

1. Keep the pinned environment unchanged while collecting the complete Conda/pip solver error and `conda info`.
2. Check that the target OS/architecture and configured Conda channels support Python 3.8 and the PyTorch 1.12.1/CUDA 11.3 stack.
3. Use the site's supported Conda/Mamba setup or ask PACE support which module/driver combination is compatible. Do not install a newer PyTorch/torchvision pair independently; they must be a compatible pair.
4. If a dependency must change, record the exact package/version, reason, environment export, and whether the inference result changed in `execution/README.md`.

### CUDA is unavailable or the job has no GPU

Check `torch.cuda.is_available()`, `torch.version.cuda`, `nvidia-smi`, the SLURM job allocation, and any site module instructions from within the allocated compute node. Confirm the correct GPU partition and resource-request syntax with ICE documentation/support. Do not diagnose GPU availability from a login node.

### The checkpoint cannot be downloaded or loaded

Check that the download is complete, file size is 432,419,871 bytes, and SHA-256 is `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`. A changed digest means do not trust/use that file; download again from the pinned Hugging Face revision. For load errors, confirm the ClimaX source commit matches this guide, use the pinned environment, and retain the full stack trace. The upstream loader maps channel embeddings to variable embeddings and interpolates positional embeddings for the forecast grid; avoid manually renaming checkpoint keys or silently ignoring load warnings.

### Input variable, shape, or normalization errors

Confirm the input directory follows the upstream preprocessor output structure and was generated by the pinned ClimaX preprocessing script. Every configured input and output variable must be present in the selected shard and in both normalization files; lat/lon must match the 32 x 64 grid; and the selected timestep plus lead must fit inside the shard. Do not edit variable ordering to silence a shape mismatch—the model's variable IDs depend on the config order.

### Out-of-memory or wall-time limit

The provided script runs batch size one. Verify that only one SLURM task/GPU is allocated, note the GPU model and available memory, and reduce unrelated processes or request a larger-memory GPU if available. Record any resource-request changes and failed job ID. Avoid changing model precision until the baseline run is reproducible.

## 9. Cleanup

Keep source, checkpoint, and preprocessed ERA5 data if you plan to rerun; they are inputs, not temporary output. After copying required evidence to an approved location, remove only the specific run output or failed temporary download that you no longer need:

```bash
rm -rf -- "$OUTPUT_DIR"
# Optional, only after confirming this checkpoint is no longer needed:
# rm -f -- "$CHECKPOINT"
```

To remove the Conda environment after preserving its explicit package list:

```bash
conda env remove -n climax-inference
```

Consult ICE storage/quota policy before removing files from shared project or scratch storage.

## Supporting references

- [ClimaX upstream repository](https://github.com/microsoft/ClimaX)
- [ClimaX usage: checkpoints, ERA5 preparation, and forecasting](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/docs/usage.md)
- [ClimaX global forecasting config](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/configs/global_forecast_climax.yaml)
- [ClimaX ERA5 preprocessing code](https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/data_preprocessing/nc2np_equally_era5.py)
- [Pinned pretrained checkpoint](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt)
- [WeatherBench ERA5 archive](https://dataserv.ub.tum.de/index.php/s/m1524895)
- [PACE ICE documentation](https://docs.pace.gatech.edu/ice/)

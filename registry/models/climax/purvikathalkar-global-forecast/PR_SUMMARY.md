Relates to issue https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/10

## Summary

Adds reproducible workflows for **frozen pretrained-encoder inference** and single-variable 72-hour forecasting with Microsoft ClimaX on ERA5 data at Georgia Tech AI Makerspace (PACE ICE). The forecast workflow accepts either the full `geopotential_5.625deg` pressure-level source or `geopotential_500_5.625deg`, prepares a one-variable dataset, and fine-tunes the pinned upstream GlobalForecast model to predict `geopotential_500`.

The previously recorded successful PACE run validates embedding extraction only. The new geopotential preprocessing/training/forecast path is documented and wired through the original ClimaX trainer and inference model, but still needs a PACE execution before its forecast metrics can be reported as empirically reproduced.

The top-level README provides the student-facing setup and reproduction guide. `workflows/climax-global-forecast/README.md` contains detailed PACE instructions, troubleshooting, and cleanup. `workflows/climax-global-forecast/ASSIGNMENTDETAILS.md` records assignment evidence, provenance, observed results, limitations, and rubric coverage.

## Important files

* [README.md](README.md): Primary guide for environment setup, model/checkpoint retrieval, WeatherBench data preparation, embedding inference, validation, and cleanup.
* [workflows/climax-global-forecast/README.md](workflows/climax-global-forecast/README.md): Detailed PACE runbook, including scratch paths, data preprocessing, inference commands, troubleshooting, and output checks.
* [workflows/climax-global-forecast/ASSIGNMENTDETAILS.md](workflows/climax-global-forecast/ASSIGNMENTDETAILS.md): Assignment evidence record, reference run, artifact provenance, validation results, known limitations, and rubric mapping.
* [workflows/climax-global-forecast/environment.yml](workflows/climax-global-forecast/environment.yml): Pinned Conda and pip dependencies, including Python 3.8.16, PyTorch 1.12.1, CUDA runtime 11.3.1, pip 24.0, and MKL 2024.0.
* [workflows/climax-global-forecast/scripts/setup_env.sh](workflows/climax-global-forecast/scripts/setup_env.sh): Initializes Miniforge, creates or updates the scratch environment with Mamba (Conda fallback), applies compatibility pins, checks dependencies/CUDA, and captures package inventories.
* [workflows/climax-global-forecast/scripts/embed.py](workflows/climax-global-forecast/scripts/embed.py): Verifies the pretrained checkpoint, strictly loads the pinned ClimaX encoder (including a narrowly scoped legacy checkpoint-key mapping), extracts spatial and pooled embeddings, and records metrics/provenance.
* [workflows/climax-global-forecast/scripts/prepare_geopotential.py](workflows/climax-global-forecast/scripts/prepare_geopotential.py): Converts either pressure-level or single-level 5.625-degree geopotential NetCDF data into normalized train/validation/test shards for forecasting.
* [workflows/climax-global-forecast/tests/test_prepare_geopotential.py](workflows/climax-global-forecast/tests/test_prepare_geopotential.py): Exercises both NetCDF source layouts and verifies the prepared shard/statistics/climatology schema.
* [workflows/climax-global-forecast/scripts/infer.py](workflows/climax-global-forecast/scripts/infer.py): Supports selecting a subset of configured inputs/outputs, including geopotential-only predictions from a fine-tuned GlobalForecast checkpoint.
* [workflows/climax-global-forecast/scripts/validate_embeddings.py](workflows/climax-global-forecast/scripts/validate_embeddings.py): Checks the saved NPZ against its metrics and compares arrays from an independent repeat run.
* [workflows/climax-global-forecast/slurm/embed.sbatch](workflows/climax-global-forecast/slurm/embed.sbatch): Primary GPU SLURM job for frozen embedding inference.
* [workflows/climax-global-forecast/logs/](workflows/climax-global-forecast/logs/): Sanitized SLURM stdout/stderr for successful and failed attempts, including jobs `6209971` and `6209978`.
* [results/embeddings-6209978/metrics.json](results/embeddings-6209978/metrics.json) and [results/embeddings-6209978/climax-embeddings.npz](results/embeddings-6209978/climax-embeddings.npz): Committed machine-readable run results and generated embedding artifact.
* [workflows/climax-global-forecast/execution/](workflows/climax-global-forecast/execution/): Captured explicit Conda and pip package inventories.
* [workflows/climax-global-forecast/slurm/finetune.sbatch](workflows/climax-global-forecast/slurm/finetune.sbatch) and [workflows/climax-global-forecast/slurm/infer.sbatch](workflows/climax-global-forecast/slurm/infer.sbatch): Upstream fine-tuning and forecast inference jobs; `FORECAST_VARIABLE=geopotential_500` opts into the reduced single-variable task.

## Reproduction target and provenance

* **Model code:** Microsoft ClimaX, commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`.
* **Configuration:** Upstream `configs/pretrain_climax.yaml` at the pinned ClimaX commit.
* **Pretrained checkpoint:** `tungnd/climax`, Hugging Face revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4`, file `5.625deg.ckpt`; SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43`.
* **Input source:** TUM WeatherBench ERA5 5.625-degree archive; the workflow preprocesses the parent `geopotential` pressure-level group and passes normalized `geopotential_500` to the encoder.
* **Recorded input:** `geopotential_npz_2016_test/test/2016_0.npz`; SHA-256 `6ab0ed6c75b6a6b7bd83fe4423ee2d5f1ac55d36d6a124e39cd571a0f8d99567`.
* **Successful split:** 2016 test data, 32 samples with indices `0, 24, 48, …, 744`.
* Large model weights and ERA5 source data are retrieved externally; their sources, identifiers, verification hashes, and retrieval steps are documented.

## Successful PACE run

SLURM job `6209978` (`climax-embed`) completed with state `COMPLETED` and exit code `0:0`.

* Account / partition / QOS: `coc` / `ice-gpu` / `coc-ice`.
* Compute node: `atl1-1-02-009-32-0`.
* Resources: 1 × NVIDIA Tesla V100-PCIE-16GB, 4 CPUs, 32768 MiB host memory; one-hour job limit.
* Output directory: `~/scratch/climax-reproduction/runs/embeddings-6209978/`.
* The run log records ClimaX commit, checkpoint/config/input hashes, Python/PyTorch/CUDA versions, output path, and result metrics.

## Validation

Smoke checks passed:

* Spatial embedding shape: `(32, 512, 1024)`.
* Pooled embedding shape: `(32, 1024)`.
* All embedding values finite: `True`.
* Representation nonconstant: `True`.
* Repeated inference within the job: deterministic within the configured tolerance.
* NPZ archive integrity: passed; output SHA-256 matches `metrics.json`.

Independent jobs `6209971` and `6209978` recorded the same output SHA-256:

`7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158`

This is byte-for-byte reproducibility evidence for the generated embedding NPZ under the recorded conditions. It is a technical validation of the encoder output, not a forecast-skill metric. No forecast RMSE or published benchmark result is claimed until the new forecasting workflow has been run.

## Reproducibility improvements and troubleshooting

* Resolved peer feedback by temporarily disabling Bash `nounset` around Conda hooks/activation, requesting the V100 compatible with the pinned PyTorch/CUDA stack, and clarifying required preprocessing.
* Added a converter for both pressure-level `geopotential_5.625deg` and single-level `geopotential_500_5.625deg` inputs, plus one-variable fine-tuning and inference overrides.
* Documented Miniforge initialization in fresh PACE shells and Mamba-first environment setup with Conda fallback.
* Pinned required software and directed environment, package caches, data, checkpoint, and run outputs to scratch.
* Added checkpoint digest verification and strict loading. The legacy `channel_*` to `var_*` parameter rename is explicitly mapped; `strict=False` is not used.
* Added output-shape, finiteness, nonconstant, deterministic-repeat, checksum, and independent-run validation.
* Preserved earlier job logs: an initial wrong data path had no test shards; a checkpoint attempt exposed renamed parameter keys; early repeat checks required an explicit numerical tolerance.

## Known limitations

* The successful run uses the 2016 test shard only. A 2017 NetCDF file reportedly failed with an HDF read error during data preparation; its original traceback is not in the checked-in logs. The successful run is not a full-period or full-benchmark evaluation.
* The added forecast workflow has not yet been executed on PACE. Its expected shape/metrics and exact training/inference commands are documented, but no forecast RMSE or skill claim is made until a run is recorded.
* Fine-tuned forecast checkpoints are task-specific. The pretrained `5.625deg.ckpt` initializes training but cannot be used directly for forecast inference; any original-repository checkpoint must load strictly with the pinned GlobalForecast architecture.
* The pinned Python/PyTorch/CUDA stack is older. The documented CUDA check should be performed on the actual target allocation; this recorded run used a V100.
* Datasets and pretrained checkpoint weights are downloaded separately rather than stored in Git.

## Validation performed

* PACE batch inference completed successfully, and the run outputs are recorded.
* `python -m unittest discover -s workflows/climax-global-forecast/tests -v` passed five synthetic checks for pressure-level selection, single-level input, rejected missing 500 hPa, generated shard/climatology shapes, and normalization stats.
* Shell syntax and `git diff --check` passed.
* The new preprocessing and forecast scripts still require an end-to-end PACE run, including verification for both NetCDF source layouts, before the contribution can claim forecast-path validation.
* The saved NPZ passed archive-integrity testing; its SHA-256 matches `metrics.json` and the independent run log.
* Python syntax/editor checks passed for the inference and validation scripts.
* Shell syntax checks passed for environment setup and SLURM scripts.
* Local Markdown links and `git diff --check` passed.

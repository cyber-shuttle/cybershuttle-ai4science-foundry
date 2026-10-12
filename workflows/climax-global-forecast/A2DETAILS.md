# Reproducible ClimaX inference — assignment details

This file records the assignment scope, artifact provenance, executed PACE workflow, observed results, limitations, and rubric coverage. The primary workflow is **frozen pretrained-encoder inference**: it extracts ClimaX representations from ERA5 and does not fine-tune the model or produce weather forecasts.

The [top-level README](../../README.md) is the student-facing setup-to-validation guide. The [detailed workflow README](README.md) provides full PACE instructions, troubleshooting, and cleanup. This file complements those guides with a concise evidence record.

> **Scope clarification for reviewers:** The initial proposal and some optional files describe fine-tuning ClimaX for 72-hour weather forecasts. That experiment is not the successful reproduction reported here. The completed assignment path uses pretrained weights directly to extract embeddings, per the no-fine-tuning objective. Do not interpret these embeddings as forecasts or as evidence of forecast skill.

## 1. Project overview and reproduction target

- **Model:** Microsoft ClimaX, using its pretrained encoder.
- **Task:** encode ERA5 `geopotential_500` fields on the WeatherBench 5.625-degree grid.
- **Inference mode:** frozen pretrained checkpoint, no training stage.
- **Input batch:** 32 samples from the 2016 test shard, with sample indices `0, 24, 48, …, 744`.
- **Input tensor shape:** `(32, 32, 64)` before adding the encoder's channel dimension.
- **Outputs:** spatial patch embeddings `(32, 512, 1024)` and mean-pooled embeddings `(32, 1024)`.
- **Interpretation:** embeddings are latent representations of input fields, not future atmospheric fields.

## 2. Software environment and PACE setup

The environment is specified in [`environment.yml`](environment.yml). Key observed and pinned versions are:

| Component | Version |
|---|---:|
| Python | 3.8.16 |
| PyTorch | 1.12.1 |
| PyTorch CUDA runtime | 11.3 / pinned cudatoolkit 11.3.1 |
| torchvision | 0.13.1 |
| PyTorch Lightning / lightning-lite | 1.8.0 |
| pip | 24.0 |
| MKL | 2024.0 |
| timm | 0.6.12 |
| NumPy | 1.23.5 |

The environment file also pins the remaining data, model, Lightning, logging, and utility dependencies. [`execution/conda-explicit.txt`](execution/conda-explicit.txt) and [`execution/pip-freeze.txt`](execution/pip-freeze.txt) preserve the captured package inventories. [`scripts/setup_env.sh`](scripts/setup_env.sh) initializes Miniforge, prefers Mamba (Conda fallback), creates or updates the environment under scratch, reapplies the pip/MKL compatibility pins, checks imports, and attempts a CUDA tensor operation.

From an ICE login, initialize Miniforge and request an interactive GPU shell for environment checks or manual embedding inference:

```bash
module purge
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
srun --account="<YOUR_PACE_ACCOUNT>" --partition=ice-gpu --gres=gpu:1 \
  --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```

Inside the allocation, follow [README steps 1–3](../../README.md) to clone the assignment branch, configure scratch storage, run `bash scripts/setup_env.sh`, and retrieve ClimaX and its checkpoint. Verify the environment with:

```bash
conda activate climax-inference
python --version
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

Expected: Python `3.8.16`, PyTorch `1.12.1`, CUDA runtime `11.3.1`, and `CUDA available: True` inside an allocated GPU shell. The successful inference was run as a batch job; its own actual resource allocation is recorded in section 6, not inferred from this interactive-shell example.

## 3. Model, checkpoint, data, and configuration provenance

| Artifact | Source and identifier | Recorded verification |
|---|---|---|
| Workflow repository | [`purvikathalkar/climax-inference`](https://github.com/purvikathalkar/climax-inference), branch `climax-inference` | Workflow commit recorded by job: `063697e0d18e615c3d662fc7c0076b6c3edf801c` |
| ClimaX implementation | [`microsoft/ClimaX`](https://github.com/microsoft/ClimaX), commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c` | Commit also recorded in the job log |
| Model configuration | `configs/pretrain_climax.yaml` at the ClimaX commit | SHA-256 `38ad0714eff479c4cfb80bb32997041f4af5d7308273e7748beb741b208d2e27` |
| Pretrained checkpoint | [`tungnd/climax/5.625deg.ckpt`](https://huggingface.co/tungnd/climax/resolve/6f9d4a5c0c2602965b2a381e60636fed7f12b8f4/5.625deg.ckpt), revision `6f9d4a5c0c2602965b2a381e60636fed7f12b8f4` | 432,419,871 bytes; SHA-256 `6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43` |
| WeatherBench data | [TUM WeatherBench archive](https://dataserv.ub.tum.de/index.php/s/m1524895), ERA5 5.625-degree `geopotential` pressure-level group and `constants` | Retrieval and unpacking steps are in [README step 4](../../README.md) |
| Prepared test input | `geopotential_npz_2016_test/test/2016_0.npz` | SHA-256 `6ab0ed6c75b6a6b7bd83fe4423ee2d5f1ac55d36d6a124e39cd571a0f8d99567` |
| Normalization | `normalize_mean.npz`, `normalize_std.npz` from the prepared ERA5 data | SHA-256 values in the job metrics/log |
| Output | `results/embeddings-6209978/climax-embeddings.npz` | SHA-256 `7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158` |

The checkpoint is downloaded from the pinned Hugging Face revision using the `curl` command in [README step 3](../../README.md); the next step verifies the expected SHA-256. The raw data is retrieved from the TUM archive rather than committed. The primary path uses the parent pressure-level `geopotential` group plus constants because that is the layout expected by the pinned ClimaX preprocessing code; it passes only the normalized `geopotential_500` channel to the encoder.

The documented reduced preprocessing split uses 2013–2014 for training statistics, 2015 for validation, and 2016 for testing (`--end_year 2017`, exclusive), with eight shards. A 2017 NetCDF input reportedly encountered an HDF read error and was excluded. Therefore, the reported result is specifically a 2016 test-shard pilot, not a full-period or full-benchmark evaluation. The original HDF traceback is not present in the checked-in logs.

## 4. Inference workflow and compute request

The primary implementation is [`scripts/embed.py`](scripts/embed.py), submitted using [`slurm/embed.sbatch`](slurm/embed.sbatch). The batch script selects the prepared data directory, pretrained checkpoint, pinned ClimaX source/configuration, 32 samples, stride 24, and CUDA device. It writes an NPZ and `metrics.json` under:

```text
$HOME/scratch/climax-reproduction/runs/embeddings-<job-id>/
```

Submit from the workflow directory after creating the log directory:

```bash
cd "$HOME/scratch/climax-inference/workflows/climax-global-forecast"
mkdir -p logs
sbatch --account="<YOUR_PACE_ACCOUNT>" slurm/embed.sbatch
squeue --me
```

The successful batch job's script requested one GPU, 4 CPUs, 32 GiB host memory, and a one-hour SLURM time limit. This differs from the smaller 16 GiB interactive request documented for environment setup; the two requests serve different purposes. The job used account `coc` and partition `ice-gpu`.

To reproduce with a changed job ID, ensure the following inputs exist at the paths used by the batch script: the `climax-inference` checkout under scratch, the ClimaX source at the pinned commit, the checkpoint at the documented path, the preprocessed 2016 test split at `data/geopotential_npz_2016_test`, and an activated `climax-inference` environment. Full download, preprocessing, and output-inspection commands are in [README steps 3–6](../../README.md).

## 5. Output files and execution evidence

Successful PACE job `6209978` (`climax-embed`) completed with state `COMPLETED` and exit code `0:0`.

| Run detail | Observed |
|---|---|
| Account / partition / QOS | `coc` / `ice-gpu` / `coc-ice` |
| Node | `atl1-1-02-009-32-0` |
| GPU | 1 × NVIDIA Tesla V100-PCIE-16GB |
| CPUs | 4 |
| Host memory reported by SLURM | 32768 MiB |
| SLURM output location | `/storage/ice1/9/2/pkathalkar3/climax-inference/workflows/climax-global-forecast` |
| Successful run directory | `~/scratch/climax-reproduction/runs/embeddings-6209978/` |

Repository evidence:

- [`logs/climax-embed-6209978.out`](logs/climax-embed-6209978.out): sanitized SLURM stdout, node/GPU information, source and artifact provenance, and recorded smoke-test metrics.
- [`results/embeddings-6209978/metrics.json`](../../results/embeddings-6209978/metrics.json): machine-readable results, expected shapes, validation flags, input/config/checkpoint hashes, and output hash.
- [`results/embeddings-6209978/climax-embeddings.npz`](../../results/embeddings-6209978/climax-embeddings.npz): generated embedding arrays; approximately 62.7 MB.
- [`logs/climax-embed-6209971.out`](logs/climax-embed-6209971.out): independent successful run log with the same output digest.

The NPZ and metrics are committed as artifacts; the larger source dataset and checkpoint are external downloads. Conda and pip inventories are available under [`execution/`](execution/).

## 6. Validation: expected and observed

### Smoke and integrity checks

The embedding job checks that strict checkpoint loading succeeds, the expected spatial and pooled dimensions are returned, the values are finite and nonconstant, and repeated inference within the job is deterministic.

| Check | Expected | Observed |
|---|---|---|
| Strict checkpoint load | No missing/unexpected state keys after the documented legacy-name mapping | Passed |
| Spatial embedding shape | `(32, 512, 1024)` | `(32, 512, 1024)` |
| Pooled embedding shape | `(32, 1024)` | `(32, 1024)` |
| All embedding values finite | `True` | `True` |
| Representation nonconstant | `True` | `True` |
| Repeated inference within job | Equal within `rtol=1e-5`, `atol=1e-6` | Passed |
| Mean token standard deviation | Positive | `0.25244829058647156` |
| Mean pooled-vector L2 norm | Positive | `7.482089042663574` |
| Output archive checksum | Equal to `metrics.json` | `7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158` |
| NPZ archive integrity | All compressed arrays readable | Passed locally |

### Independent repeat check

Jobs `6209971` and `6209978` used the same recorded checkpoint, config, input shard, sample indices, and output settings. Their separate output paths have the same SHA-256:

```text
6209971  7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158
6209978  7965fd2362e4471d1a5bc92f32da040ced27ee5e5f24ae910fe22c5adfcf2158
```

This digest equality is a byte-for-byte independent reproduction check, stronger than comparing only tensor dimensions. The second NPZ is not committed; its digest and provenance are preserved in its job log. For a future direct array-by-array comparison when both NPZs are available, use [`scripts/validate_embeddings.py`](scripts/validate_embeddings.py):

```bash
python scripts/validate_embeddings.py \
  "$HOME/scratch/climax-reproduction/runs/embeddings-6209978/climax-embeddings.npz" \
  --metrics "$HOME/scratch/climax-reproduction/runs/embeddings-6209978/metrics.json" \
  --compare "$HOME/scratch/climax-reproduction/runs/embeddings-<repeat-job-id>/climax-embeddings.npz"
```

### Scientific interpretation and limit of the evidence

The checks support that the pinned encoder accepted the documented ERA5 input and produced finite, non-degenerate, exactly repeatable representations for the sampled 2016 shard. This provides technical and computational reproducibility evidence. The workflow does **not** compute a forecast, compare predictions with targets, train a downstream probe, or establish that the embeddings improve a scientific prediction task. No forecast RMSE or published benchmark score is claimed. A downstream linear-probe experiment or forecast evaluation would be a separate extension, not an observed result of this assignment run.

## 7. Failures, troubleshooting, and deviations

Earlier attempts are retained in [`logs/`](logs/):

- `6209668`: no `.npz` test shards found under the initial data path. The job script was corrected to use `geopotential_npz_2016_test`.
- `6209869`: strict checkpoint load exposed legacy `channel_*` keys versus current `var_*` model keys. [`embed.py`](scripts/embed.py) now maps only those known names and still loads with `strict=True`.
- `6209904` and `6209926`: repeated forward passes initially failed a strict equality-style check. The check was corrected to use numeric tolerances.
- `6209961`: a repeat comparison exceeded the initial tolerance slightly; the final numerical check records and applies `rtol=1e-5`, `atol=1e-6`.
- Data preparation: the reported 2017 HDF read error led to the successful pilot using the available 2016 test shard. The source traceback was not captured in the checked-in logs.

For new attempts, inspect the job's `.out` and `.err`, verify the data path and expected shard, check the checkpoint digest, and confirm the allocated node sees a CUDA device. Do not suppress checkpoint errors with `strict=False`, silently switch to CPU, or describe the 2016 pilot as full-period evaluation. The [workflow troubleshooting section](README.md#11-troubleshooting) covers dependency, CUDA, checkpoint, input-shape, and resource issues.

## 8. Reproduction and cleanup

Follow the [top-level README](../../README.md) in sequence: PACE access and scratch setup; Miniforge and environment setup; pinned ClimaX source and checkpoint retrieval; WeatherBench download and reduced preprocessing; SLURM inference; artifact validation. Keep the model source, package caches, raw data, checkpoint, and run outputs under scratch. Preserve the job log, package inventories, metrics, and output digest as evidence.

After copying any evidence needed for evaluation, use the targeted cleanup commands in [workflow README section 12](README.md#12-cleanup). Do not remove shared or unrelated files, and do not commit the full raw ERA5 dataset or pretrained checkpoint.

## 9. Rubric coverage and outstanding scope

| Assignment criterion | Evidence in this submission | Status / boundary |
|---|---|---|
| Environment & Setup — 15 pts | Pinned `environment.yml`, setup helper, expected version/CUDA check, Conda and pip inventories | Documented and used for successful inference |
| Model, Data & Artifact Traceability — 15 pts | ClimaX commit, checkpoint revision/hash, config hash, WeatherBench source, input shard/hash, normalization hashes, retrieval instructions | Traceable for the recorded 2016 input |
| Inference Workflow & Job Scripts — 25 pts | `embed.py`, `embed.sbatch`, exact sample/variable settings, resource requests, expected outputs | Successful frozen-encoder workflow; not a fine-tuned forecast run |
| Execution Evidence & Logs — 15 pts | Completed job record, sanitized SLURM log, committed metrics and NPZ, package inventories, earlier failure logs | Successful run is evidenced; original 2017 HDF traceback is unavailable |
| Validation of Results — 20 pts | Shape/finite/nonconstant checks, deterministic inference, NPZ integrity/hash, byte-identical output hashes from two separate jobs | Strong technical reproducibility; no downstream scientific skill score |
| Documentation & Reproducibility — 10 pts | Top-level guide, detailed PACE guide, this evidence record, troubleshooting and cleanup | Documents the successful pilot and limitations |

Related tracking issue: [cybershuttle-ai4science-foundry issue #10](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/10).

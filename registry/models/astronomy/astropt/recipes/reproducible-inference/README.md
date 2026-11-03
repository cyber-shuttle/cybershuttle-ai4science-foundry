# AstroPT reproducible inference on Georgia Tech ICE

This repository reproduces GPU inference with the 95M-parameter AstroPT v2.0 astronomy foundation model. It includes a fast smoke test, a separate scientific validation, the SLURM submission script, a locked software environment, machine-readable results, and complete logs from one successful and one failed job.

The successful reference run was SLURM job `6133484` on an NVIDIA H100 80GB HBM3 GPU. The failed V100 run (`6133445`) is intentionally retained as troubleshooting evidence.

## Provenance

| Item | Exact source or version |
| --- | --- |
| Model code | [Smith42/astroPT](https://github.com/Smith42/astroPT), PyPI package `astropt==2.0.5` |
| Upstream source commit inspected | [`fa361ae5b6631c89525429755ed3fc2b2d304eca`](https://github.com/Smith42/astroPT/commit/fa361ae5b6631c89525429755ed3fc2b2d304eca) |
| Model repository | [Smith42/astroPT_v2.0](https://huggingface.co/Smith42/astroPT_v2.0) |
| Checkpoint directory / weights | `astropt/095M` / `ckpt.pt` |
| Parameter count | 95,177,472 |
| Input dataset | [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies), revision `v2.0`, split `test` |
| Input fields | `image` and non-null `mag_g` |
| Tracking issue | [cyber-shuttle/cybershuttle-ai4science-foundry#2](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/2) |

The scripts use Hugging Face streaming, so the full dataset is not downloaded. `load_astropt` downloads the released checkpoint into the Hugging Face cache. The source commit above is an inspected upstream reference; the installed runtime package is identified independently by its released version, `2.0.5`.

## Repository contents

| Path | Purpose |
| --- | --- |
| `scripts/smoke_load.py` | Confirms the package, CUDA device, checkpoint load, and parameter count. |
| `scripts/inference_small.py` | Runs the 8-example smoke inference and checks that all embeddings are finite. |
| `scripts/validate_mag_g.py` | Runs the 1,000-galaxy scientific validation and linear probe. |
| `slurm/astropt_inference.slurm` | Requests the GPU resources and runs all three scripts in order. |
| `environment.yml` | Recreates the complete Conda and pip environment. |
| `requirements-lock.txt` | Records the exact Python package set as a portable pip lock. |
| `outputs/*.json` | Machine-readable smoke and validation metrics. |
| `outputs/mag_g_probe.png` | Held-out true-versus-predicted validation plot. |
| `logs/astropt_a2_6133484.*` | Complete stdout/stderr for the successful H100 run. |
| `logs/astropt_a2_6133445.*` | Complete stdout/stderr for the failed V100 run. |

The three short `logs/*.log` files are auxiliary standalone runs. The authoritative reported metrics below come from the complete successful SLURM log and its JSON outputs.

## ICE setup

Run these commands after copying or cloning this repository into a storage location with adequate quota (scratch is recommended). No user-specific absolute path is required.

Because creating the Conda environment can be resource-intensive, use an interactive ICE CPU allocation rather than running `conda env create` on a shared login node. Replace `YOUR_ACCOUNT` and `YOUR_QOS` with values authorized for your account.

```bash
salloc -A YOUR_ACCOUNT -q YOUR_QOS -p ice-cpu -N 1 --cpus-per-task=4 --mem=32G -t 01:00:00
cd registry/models/astronomy/astropt/recipes/reproducible-inference
module purge
module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda env create --prefix "$PWD/env" --file environment.yml
conda activate "$PWD/env"
mkdir -p logs outputs hf_cache
export HF_HOME="$PWD/hf_cache"
```

The reference environment uses Python 3.10.22, AstroPT 2.0.5, and PyTorch 2.14.1+cu130. Confirm the core versions after installation:

```bash
python - <<'PY'
import importlib.metadata
import sys
import torch

print("Python:", sys.version.split()[0])
print("AstroPT:", importlib.metadata.version("astropt"))
print("PyTorch:", torch.__version__)
PY
```

`environment.yml` is the primary installation specification. `requirements-lock.txt` is a second, exact record of the pip packages and can be used inside another Python 3.10 environment with `python -m pip install -r requirements-lock.txt`.

The SLURM file records the allocation used for the verified run: account `ece`, QOS `coc-ice`, partition `ice-gpu`, one H100 GPU, four CPUs, 32 GB RAM, and a 30-minute limit. ICE users without access to that account or QOS must replace only the `--account` and `--qos` values with their authorized allocation. Keep the H100 request because the locked CUDA 13.0 PyTorch build does not support the V100's compute capability 7.0. If the recorded Miniforge module is no longer available, check the installed versions with `module spider miniforge` before changing the module line.

An optional Hugging Face token avoids anonymous-download rate limits:

```bash
export HF_TOKEN="your_read_token"
```

Do not place a token in this repository or in the SLURM script.

## Execute the complete workflow

Submit from the repository root; the script uses the SLURM submission directory, activates `./env`, and sets the cache to `./hf_cache`.

If your shell prompt shows an active Conda environment such as `(base)`, run `conda deactivate` before submitting. This avoids an activation conflict with the job's `set -u` setting.

```bash
conda deactivate  # only if a Conda environment such as (base) is active
JOB_ID=$(sbatch --parsable slurm/astropt_inference.slurm)
echo "Submitted job: $JOB_ID"
squeue -j "$JOB_ID"
```

After the job leaves the queue, inspect its status, resources, and logs:

```bash
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,AllocTRES,MaxRSS
less "logs/astropt_a2_${JOB_ID}.out"
less "logs/astropt_a2_${JOB_ID}.err"
```

A successful run ends with `WORKFLOW COMPLETED SUCCESSFULLY` in stdout. Because the script uses `set -euo pipefail`, any failed command stops the workflow and gives the job a nonzero exit status.

## Smoke test

The smoke phase is deliberately small and is not the scientific validation. It first loads the model (`smoke_load.py`), then processes eight dataset examples in one batch (`inference_small.py`). The acceptance checks are:

- checkpoint loads on CUDA;
- parameter count is 95,177,472;
- output shape is `(8, 768)`; and
- every embedding value is finite.

Reference H100 result from job `6133484`: 0.5926 seconds for inference and 0.534 GB peak CUDA memory. The output metadata is in `outputs/inference_results.json`; the small arrays are regeneratable and ignored by Git.

To run only this phase, first obtain an H100 GPU allocation, activate the environment, and run:

```bash
python scripts/smoke_load.py
python scripts/inference_small.py
```

## Substantive scientific validation

`validate_mag_g.py` streams the first 1,000 test galaxies with non-null `mag_g`, generates 768-dimensional embeddings in batches of 32, fits ordinary least squares on the first 500 examples, and evaluates on the held-out final 500. It compares the probe against a baseline that always predicts the training-set mean. This is separate from the smoke test and tests whether the learned image representation contains predictive information about g-band magnitude.

Reference H100 result from job `6133484`:

| Check | Linear probe | Mean baseline |
| --- | ---: | ---: |
| MSE | 0.965233 | 1.160485 |
| MAE | 0.731518 | 0.772794 |
| R² | 0.167680 | -0.000686 |

The probe beats the mean baseline on both MSE and MAE. Embedding shape was `(1000, 768)`, embedding runtime was 46.50 seconds, and peak CUDA memory was 0.978 GB. Exact values are in `outputs/validation_mag_g.json`; the diagnostic plot is `outputs/mag_g_probe.png`.

`mag_g` is the galaxy's apparent magnitude in the g optical band, so it represents a physically meaningful observable related to how bright the galaxy appears in that wavelength range. Because AstroPT is pretrained directly on galaxy images, a useful representation should preserve information related to observable galaxy properties such as brightness. The linear probe's improvement over the mean baseline therefore indicates that the AstroPT embeddings retain meaningful astronomical information rather than simply producing numerically valid vectors. The R² of 0.168 is modest, so this result should be interpreted as evidence that the embeddings contain recoverable information about `mag_g`, not that they fully determine the galaxy's magnitude.

To run only the validation in an activated H100 allocation:

```bash
python scripts/validate_mag_g.py
```

## Verify a reproduced run

From the repository root, this checks the required smoke and scientific-validation conditions without requiring exact floating-point equality with the reference run:

```bash
python - <<'PY'
import json
from pathlib import Path

smoke = json.loads(Path("outputs/inference_results.json").read_text())
valid = json.loads(Path("outputs/validation_mag_g.json").read_text())

assert smoke["num_examples"] == 8
assert smoke["embedding_shape"] == [8, 768]
assert smoke["all_embeddings_finite"] is True
assert valid["num_examples"] == 1000
assert valid["train_examples"] == valid["test_examples"] == 500
assert valid["embedding_dimension"] == 768
assert valid["linear_probe"]["mse"] < valid["mean_baseline"]["mse"]
assert valid["linear_probe"]["mae"] < valid["mean_baseline"]["mae"]
print("Smoke test and substantive validation passed.")
PY
```

## Execution evidence

Successful job `6133484` used an NVIDIA H100 80GB HBM3 GPU, four CPUs, and 32 GB requested RAM. The SLURM epilog reports 1:30 wall time, 6:00 CPU time, and 3,392,784 KiB maximum resident memory. Its stdout contains the complete environment, parameters, results, and completion marker; stderr contains only the documented activation, anonymous-Hub, and dataset formatting warnings.

Failed job `6133445` ran on a Tesla V100 (compute capability 7.0) and stopped during the small inference with `CUBLAS_STATUS_ARCH_MISMATCH`. PyTorch 2.14.1+cu130 was built for compute capability 7.5 or newer. The failure is preserved in `logs/astropt_a2_6133445.out` and `logs/astropt_a2_6133445.err`; the corrected SLURM request explicitly selects an H100.

## Troubleshooting

- `CUBLAS_STATUS_ARCH_MISMATCH` or a compute-capability warning: verify `nvidia-smi` reports an H100 and that the job used `--gres=gpu:h100:1`. Do not treat a successful checkpoint load as a successful inference; the V100 failure occurred on the first GPU linear operation.
- `Module miniforge/... not found`: run `module spider miniforge` and use an available ICE module, then record that change with the reproduced results.
- Conda activation prints a mamba initialization warning: the preserved runs show this warning before successful activation; the explicit `conda shell.bash hook` is what the job uses.
- Hugging Face warns about anonymous requests or rate limits: export `HF_TOKEN` in the submitting shell and resubmit.
- Download or quota failure: place the repository and `hf_cache` on ICE scratch space with sufficient quota; the model and streamed dataset shards are cached there.
- A job stops without the completion marker: inspect both `%j.out` and `%j.err`, then use `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS`.

## Cleanup

The environment, Hugging Face cache, model weights, Python caches, and large generated arrays are excluded by `.gitignore`. From the repository root, remove regeneratable local state while keeping the committed JSON, plot, scripts, and execution logs:

```bash
test -f slurm/astropt_inference.slurm
conda deactivate
conda env remove --prefix "$PWD/env"
rm -rf "$PWD/hf_cache"
rm -f "$PWD/outputs/embeddings_small.npy" \
  "$PWD/outputs/mag_g_small.npy" \
  "$PWD/outputs/validation_embeddings.npy" \
  "$PWD/outputs/validation_mag_g.npy"
```

Do not remove `logs/astropt_a2_6133484.*`, `logs/astropt_a2_6133445.*`, the JSON results, or the validation plot; they are the retained execution evidence.

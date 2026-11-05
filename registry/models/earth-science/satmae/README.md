# SatMAE — reproducible inference on PACE ICE

This README covers one task: running the upstream-released, already
fine-tuned [SatMAE](https://github.com/sustainlab-group/SatMAE) multi-spectral
ViT-Base checkpoint on the fMoW-Sentinel **validation** split, reproducibly,
on Georgia Tech's PACE ICE cluster (`coc-gpu` partition). It checks the result
against the upstream-reported 62.65% top-1. Everything here was run and
verified on that cluster, not assumed from the upstream docs. See
[Troubleshooting](#troubleshooting) for the gaps found along the way.

Full job logs, `metrics.json`, confusion matrices, the resolved conda lock
file, and a job-by-job run log are in the project repo:
**https://github.com/DevP1130/satmae-reproducible-inference**

Model metadata: [`model.yaml`](model.yaml) · Execution recipe:
[`../../../../recipes/satmae-inference-classification.yaml`](../../../../recipes/satmae-inference-classification.yaml) ·
Resource profile: [`../../../../resource-profiles/pace-ice/coc-gpu.yaml`](../../../../resource-profiles/pace-ice/coc-gpu.yaml)

## Setup

All jobs are submitted **from the foundry repo root**: the sbatch scripts
resolve paths from `$SLURM_SUBMIT_DIR`. Slurm also needs `logs/` to exist
before it can write `#SBATCH --output` files.

Data, weights, the upstream SatMAE checkout, and results go under
`$SATMAE_WORKDIR` (default `~/scratch/satmae`), not into this repo. The conda
env goes to `~/scratch/envs/satmae`. ICE home is capped at 30 GB, and scratch
is 300 GB.

```bash
ssh login-ice.pace.gatech.edu

# Clone the fork and branch used by PR #28.
git clone --branch satmae-assignment-2 --single-branch \
  https://github.com/DevP1130/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry
mkdir -p logs

# 1. Clone upstream SatMAE @ 0b210ace… and build the conda env (CPU job on coc-cpu)
sbatch scripts/satmae_build_env.sbatch

# 2. One-time timm 0.3.2 patch (see Troubleshooting)
module load anaconda3
conda activate ~/scratch/envs/satmae
bash scripts/satmae_patch_timm.sh

# 3. Checkpoint (wget from Zenodo + sha256)
bash scripts/satmae_download_weights.sh

# 4. fMoW-Sentinel val split (~77.5 GB download; CPU job, took ~5h13m)
sbatch scripts/satmae_download_data.sbatch
rm -f ~/scratch/satmae/data/fmow-sentinel.tar.gz   # after extraction, to stay under quota

# 5. Rebuild image_path and write the CSV the evaluator reads
python scripts/satmae_prepare_csv.py --csv ~/scratch/satmae/data/val.csv --split val \
  --data-root ~/scratch/satmae/data --out ~/scratch/satmae/data/val_prepared.csv

# optional: class-stratified ~5,000-image subset for a quick cross-check
python scripts/satmae_prepare_csv.py --csv ~/scratch/satmae/data/val.csv --split val \
  --data-root ~/scratch/satmae/data --subset 5000 --out ~/scratch/satmae/data/val_subset5000.csv
```

The env pins Python 3.9, `pytorch=1.12.1`, `cudatoolkit=11.3`,
`torchvision=0.13.1` and `timm==0.3.2`, matching the checkpoint's original
training environment. See
[`scripts/satmae_environment.yml`](../../../../scripts/satmae_environment.yml).
The build job writes the fully resolved lock file and `pip freeze` to
`$SATMAE_WORKDIR/env/`.

## Quick verification before downloading imagery

Assignment 3 peer feedback suggested a lightweight first check. The existing
100-image smoke test and ~5,000-image subset are useful, but both still need
the full ~77.5 GB archive downloaded and the validation split extracted.

After Setup steps 1–3, you can check the model and fine-tuned checkpoint
without downloading imagery. Run this on an **allocated PACE compute node**,
with the SatMAE conda environment activated, from the foundry repository root:

```bash
python - <<'PY'
import torch
from scripts.satmae_evaluate import (
    SATMAE_WORKDIR,
    build_model,
    load_checkpoint,
)

model = build_model()
checkpoint = SATMAE_WORKDIR / "weights" / "finetune-vit-base-e7.pth"
result = load_checkpoint(model, checkpoint)

assert not result.missing_keys, result.missing_keys
assert not result.unexpected_keys, result.unexpected_keys

model.eval()
with torch.no_grad():
    output = model(torch.zeros(1, 10, 96, 96))

assert tuple(output.shape) == (1, 62), output.shape
print("Checkpoint-only verification: PASS")
PY
```

Expected: `Missing keys (0): []`, `Unexpected keys (0): []`, and
`Checkpoint-only verification: PASS`. This is a **checkpoint/forward-pass
sanity check**, not an image-based inference or accuracy test. This additional
command is proposed as a convenience check and has **not been separately run
or verified** as part of the original evidence. The previously verified
100-image smoke test and full validation result remain the functional
inference evidence below.

## Execution

```bash
# smoke test: first 100 images, 15-minute limit
sbatch scripts/satmae_smoke_test.sbatch

# primary result: full validation split (84,939 images)
sbatch scripts/satmae_eval_full.sbatch

# secondary: class-stratified subset (needs val_subset5000.csv from Setup step 5)
sbatch scripts/satmae_eval_subset.sbatch
```

Each job writes `metrics.json` (top-1, top-5, n_images, per-class accuracy,
runtime, GPU name, peak GPU memory) and `confusion_matrix.csv` under
`$SATMAE_WORKDIR/results/` (`smoke/`, `subset5000/`, or the top level). It
also writes `nvidia-smi` output, torch/CUDA versions, wall time, and `sacct`
output to `logs/satmae_*`.

`scripts/satmae_evaluate.py` can also be run directly on a compute node. See
the sbatch files for the exact flags (`--limit N` for a smoke-sized run,
`--random-init` for a random-weights baseline).
`scripts/satmae_eval_random.sbatch` runs that baseline, but it was **not**
run during verification.

## Verification

Pass criterion: full-val top-1 within ±1 percentage point of the upstream
62.65% (61.65–63.65%).

| Check | Expected | Observed |
| --- | --- | --- |
| Smoke test (100 images) | completes, 0 missing / 0 unexpected keys | **Pass**: job 6114790, top-1 97.00% / top-5 100.00%, 72s wall. Pipeline check only; the first 100 CSV rows aren't class-balanced. |
| **Full val top-1 (84,939 images)** | 62.65% ± 1 pt | **PASS: 62.65%** (job 6115115). Top-5 85.79%, 271.7s eval / 278s wall, ~1.84 GB peak GPU memory, 0 missing / 0 unexpected checkpoint keys. |
| Stratified subset (4,950 images) | informal cross-check | 59.45% top-1 / 83.03% top-5 (job 6114806), 46s wall. Lower because `--subset` samples ~80 images per class instead of val's natural class distribution. |
| Random-init baseline | ~1/62 ≈ 1.6% | Not run. |

Final lines of the passing full-val job (`logs/eval_full_6115115.out` in the
project repo):

```
Missing keys (0): []
Unexpected keys (0): []
n_images=84939 top1=62.65% top5=85.79% runtime=271.7s gpu=NVIDIA L40S peak_mem=1.835140096
```

Two things are checked:

1. **Clean checkpoint load.** `load_state_dict(strict=False)` reports empty
   missing and unexpected key lists for this exact config (`input_size=96`,
   `patch_size=8`, `group_c`, `nb_classes=62`). A non-empty list means an
   argument mismatch, and the model may be running with partly random weights.
2. **Accuracy matches upstream** on the full validation split, not a subset.
   The stratified subset reads ~3 points lower for distribution reasons, so
   it is not used as the pass criterion.

**Assignment 4 verification (November 4, 2026): PASS.** The checkpoint-only test was successfully executed on a PACE ICE CPU compute node. The model loaded with zero missing and zero unexpected checkpoint keys, and the forward pass completed with the expected output shape of (1, 62).

Verification log: `~/scratch/satmae/assignment4_verification.log`

Checkpoint SHA-256: `3a65b5974f2afefba4aab6fa78b878188ad97b841781325b092981815b2f96e1`

This checkpoint-only test verifies model loading and execution, not classification accuracy. The previously verified full-validation evaluation remains the accuracy evidence.

## Hardware and provenance

- **Verified hardware:** PACE ICE, partition `coc-gpu`, account `coc`, QOS
  `coc-ice`, 1x NVIDIA L40S (46068 MiB, confirmed via `nvidia-smi` in every
  GPU job), `--cpus-per-task=8`, `--mem=96G`. An A100
  (`--gres=gpu:a100:1`) is also available on this partition but was not
  tested. Do **not** request the AMD MI210 nodes on the same partition
  (ROCm-only).
- **Upstream code:** https://github.com/sustainlab-group/SatMAE at commit
  `0b210aceb37a14bbbd897110db5b104b3271d818` (`main`, 2025-08-10).
  License: CC BY-NC 4.0 (non-commercial use only).
- **Checkpoint:** `finetune-vit-base-e7.pth` from
  https://zenodo.org/record/7338613 (DOI
  [10.5281/zenodo.7338613](https://doi.org/10.5281/zenodo.7338613)).
  `scripts/satmae_download_weights.sh` generates a local SHA-256 file at
  `$SATMAE_WORKDIR/weights/finetune-vit-base-e7.pth.sha256`. This can detect
  changes to the locally downloaded file; an independently published
  reference checksum has not been recorded.
- **Data:** fMoW-Sentinel, validation split only, from
  https://purl.stanford.edu/vg497cb6002 (DOI
  [10.25740/vg497cb6002](https://doi.org/10.25740/vg497cb6002)), Version 1.
  The archive `fmow-sentinel.tar.gz` is 77,497,587,986 bytes. Extracting
  `val/` yields 92,263 `.tif` files; `val.csv` lists 84,939, all of which
  resolve after `satmae_prepare_csv.py`. The extra files are unreferenced.
  fMoW and Copernicus/Sentinel-2 terms apply. Do not redistribute the
  imagery.
- **Model config:** `vit_base_patch16`, `model_type=group_c`,
  `input_size=96`, `patch_size=8`, `dropped_bands=[0, 9, 10]`, channel
  groups `[0,1,2,6] [3,4,5,7] [8,9]`, `nb_classes=62`, `global_pool=True`,
  fp16 autocast, batch size 64.
- **Software:** Python 3.9.23, torch 1.12.1 (CUDA 11.3), torchvision
  0.13.1, timm 0.3.2 (patched), rasterio 1.3.6, pandas 1.5.3, numpy 1.23.5.
- **Seeds:** `torch.manual_seed(0)` / `np.random.seed(0)`. Evaluation is
  `model.eval()` with `torch.no_grad()` and no shuffling.
  `satmae_prepare_csv.py --subset` uses `--seed 0`.
- **Run dates:** all jobs ran 2026-10-08.
- **Resource usage:** env build 7m44s (`coc-cpu`, job 6106735); data
  download ~5h13m at ~3.9 MB/s (`coc-cpu`, job 6107078); full eval 278s wall
  with ~84 GiB host RAM used (job 6115115).

## Troubleshooting

**`timm` `container_abcs` ImportError.** `timm==0.3.2`'s
`timm/models/layers/helpers.py` does `from torch._six import container_abcs`.
That fails even under the pinned torch 1.12.1 (`ImportError: cannot import
name 'container_abcs' from 'torch._six'`, hit in the env build job 6106735's
import check). Fix: run `scripts/satmae_patch_timm.sh` once inside the env.
It rewrites the import to `import collections.abc as container_abcs`.

**Upstream `util/misc.py` `torch._six.inf` ImportError.** `util/misc.py`
does `from torch._six import inf`, which was removed in torch>=1.13. Torch is
pinned to 1.12.1. In addition, `scripts/satmae_evaluate.py` imports only
`models_vit_group_channels.py` and `util/datasets.py` and reimplements the
small top-1/top-5 helper. If you reuse more upstream training code, you
will hit this again.

**`timm` version mismatch.** The checkpoint and
`models_vit_group_channels.py` assume the old `timm==0.3.2`
`VisionTransformer` internals. Keep the pin.

**`torch.distributed.launch` / wandb.** Upstream commands launch via
`torch.distributed.launch --nproc_per_node=8`, and `main_finetune.py`
hard-codes `wandb.init(entity="mae-sentinel")`. The evaluator is
single-GPU and never imports `main_finetune.py`, so no launcher or wandb key
is needed.

**`image_path` column.** The `image_path` column in the raw `val.csv` from
the Stanford repository does not match the on-disk layout. Always run
`scripts/satmae_prepare_csv.py` first.

**OOM on the full validation set (job 6114969).** With `--mem=64G`, a
DataLoader worker was killed by SIGKILL and Slurm reported 1 `oom_kill`
event. Fix: `--mem=96G` in `scripts/satmae_eval_full.sbatch`. The rerun
(job 6115115) used ~84 GiB host RAM.

**`mkdir: Permission denied` in a Slurm job (job 6114779).** Resolving
the repo path from `${BASH_SOURCE[0]}` points at Slurm's spooled copy of the
script. The scripts use `$SLURM_SUBMIT_DIR`, so submit from the repo root.

**`MKL_INTERFACE_LAYER: unbound variable` (job 6114783).** Conda's
`libblas_mkl_activate.sh` hook trips `set -u`. Activation is wrapped in
`set +u; conda activate …; set -u`.

**`Invalid qos specification`.** This happens when submitting with
`-A coe-gpu`. The correct association is account `coc`, partition `coc-gpu`,
QOS `coc-ice`.

**Env build on the login node killed.** Login nodes enforce resource limits.
Build with `sbatch scripts/satmae_build_env.sbatch` on `coc-cpu`.

## Cleanup

```bash
rm -rf ~/scratch/satmae/data              # fMoW-Sentinel val imagery + CSVs
rm -rf ~/scratch/satmae/weights           # checkpoint
rm -rf ~/scratch/envs/satmae ~/scratch/conda_pkgs   # conda env + package cache
```

`~/scratch/satmae/SatMAE` (upstream checkout) and `~/scratch/satmae/results`
are small and can stay. Re-running `scripts/satmae_build_env.sbatch` reuses
an existing checkout. Note that `conda env create` fails if the env prefix
already exists.

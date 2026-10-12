# SatMAE Temporal ViT-Large — fMoW-Temporal inference on PACE ICE

This README covers one task: running the upstream-released, already
fine-tuned [SatMAE](https://github.com/sustainlab-group/SatMAE) temporal
ViT-Large checkpoint on the fMoW-Temporal **validation** CSV, reproducibly,
on Georgia Tech PACE ICE (`coc-gpu`). It compares the result to the
upstream-reported **79.99%** Top-1 (no TTA). Everything here was run on that
cluster on 2026-10-11, not assumed from the upstream docs. See
[Troubleshooting](#troubleshooting) for the gaps found along the way, and
[Verification](#verification) for expected vs observed.

This is **not** multi-spectral SatMAE (ViT-Base / fMoW-Sentinel / 62.65%).
The temporal recipe uses a different checkpoint, dataset, and model class.

`metrics.json` for smoke / subset / full, `pip freeze`, and failure notes
are in [`../../../../evidence/satmae-temporal/`](../../../../evidence/satmae-temporal/).
Weights and imagery stay on scratch and are not committed.

Cross-links:

- Model metadata: [`model.yaml`](model.yaml)
- Execution recipe: [`../../../../recipes/satmae-temporal-inference-classification.yaml`](../../../../recipes/satmae-temporal-inference-classification.yaml)
- Resource profile: [`../../../../resource-profiles/pace-ice/coc-gpu.yaml`](../../../../resource-profiles/pace-ice/coc-gpu.yaml)
- Environment file: [`../../../../scripts/satmae_temporal_environment.yml`](../../../../scripts/satmae_temporal_environment.yml)
- Benchmark record: [`../../../../benchmarks/results/satmae-temporal/fmow-temporal-pace-ice.yaml`](../../../../benchmarks/results/satmae-temporal/fmow-temporal-pace-ice.yaml)

## Setup

All jobs are submitted **from the Foundry repo root**. The sbatch scripts
resolve paths from `$SLURM_SUBMIT_DIR`. Slurm also needs `logs/` to exist
before it can write `#SBATCH --output` files. Do not build or evaluate on
the login node (about 1 CPU / 4 GB).

Data, weights, the upstream SatMAE checkout, and results go under
`$SATMAE_TEMPORAL_WORKDIR` (default `~/scratch/satmae-temporal`), not into
this repo. The conda env goes to `~/scratch/envs/satmae-temporal`. ICE home
is 30 GB (this account was already at 96.9%); scratch is 300 GB.

```
$SATMAE_TEMPORAL_WORKDIR/
  SatMAE/                 # pinned upstream clone
  env/                    # conda lock / pip freeze
  weights/                # finetune_fmow_temporal.pth + sha256 sidecar
  data/                   # val_62classes.csv, fmow/val/, prepared CSVs
  results/                # smoke/ subset500/ full/ metrics.json
```

```bash
ssh login-ice.pace.gatech.edu
cd ~/scratch/cybershuttle-ai4science-foundry
export SATMAE_TEMPORAL_WORKDIR="$HOME/scratch/satmae-temporal"
mkdir -p logs

# confirm GPU type before any GPU job (edit --gres= if not L40S)
sinfo -p coc-gpu -o "%P %G %N"

# 1. Clone upstream SatMAE @ 0b210ace… and build the conda env (coc-cpu)
sbatch scripts/satmae_temporal_build_env.sbatch

# 2. One-time timm / numpy dtype patch (also run by the build job)
#    conda activate ~/scratch/envs/satmae-temporal
#    bash scripts/satmae_temporal_patch_env.sh

# 3. Checkpoint (wget from Zenodo + sha256)
sbatch scripts/satmae_temporal_download_weights.sbatch

# 4. fMoW-RGB val imagery + val_62classes.csv (coc-cpu; do not download train)
sbatch scripts/satmae_temporal_download_data.sbatch

# 4b. If the data job finished before AOI crops existed (job 6192078):
#     SpaceNet S3 has *_rgb.jpg; SatMAE CSVs / temporal glob need
#     *_rgb_crop_0.jpg. This writes those chips from each scene's JSON bbox
#     and reruns prepare_csv.
sbatch scripts/satmae_temporal_crop_fmow.sbatch

# 5. Rebuild image paths (the data and crop sbatches already do this; rerun
#    if you copied the CSV in by hand)
python scripts/satmae_temporal_prepare_csv.py \
  --csv "$SATMAE_TEMPORAL_WORKDIR/data/val_62classes.csv" \
  --data-root "$SATMAE_TEMPORAL_WORKDIR/data" \
  --out "$SATMAE_TEMPORAL_WORKDIR/data/val_prepared.csv" \
  --subset-out "$SATMAE_TEMPORAL_WORKDIR/data/val_subset500.csv" \
  --subset-size 500 --seed 0
```

Metadata CSV source: [SatMAE Drive folder](https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD).
Imagery: `s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/val`.
If Google Drive is blocked on the compute node, `scp` `val_62classes.csv` to
`$SATMAE_TEMPORAL_WORKDIR/data/` and rerun step 5.

The env pins Python 3.9, `pytorch=1.12.1`, `cudatoolkit=11.3`,
`torchvision=0.13.1`, `timm==0.3.2`, and `numpy=1.23.5`. The build job writes
the resolved lock file and `pip freeze` to `$SATMAE_TEMPORAL_WORKDIR/env/`.

## Execution

```bash
# smoke test: first 100 rows, 30-minute limit
sbatch scripts/satmae_temporal_smoke_test.sbatch

# primary result: full prepared validation CSV (only if val imagery fits)
sbatch scripts/satmae_temporal_eval_full.sbatch

# secondary / fallback: class-stratified 500-row subset
sbatch scripts/satmae_temporal_eval_subset.sbatch

# optional chance baseline (~1/62); not required for the pass criterion
sbatch scripts/satmae_temporal_eval_random.sbatch
```

Each job writes `metrics.json` (top-1, top-5, n_images, per-class accuracy,
runtime, GPU name, peak GPU memory, checkpoint key counts) and
`confusion_matrix.csv` under `$SATMAE_TEMPORAL_WORKDIR/results/` (`smoke/`,
`subset500/`, `full/`, `random/`). It also writes `nvidia-smi`, torch/CUDA
versions, wall time, and `sacct` output to `logs/satmae_temporal_*`.

`scripts/satmae_temporal_evaluate.py` can also be run directly on a compute
node. See the sbatch files for the exact flags (`--limit N` for a smoke-sized
run, `--eval-csv` to score a subset while keeping the full val index for
companion timestamps, `--random-init` for an untrained baseline).
`scripts/satmae_temporal_eval_random.sbatch` exists but was **not** run.

## Verification

Upstream target: full-val Top-1 within ±1 point of **79.99%** (no TTA).
That match **did not happen**. The assignment asks for a smoke test, one
substantive check, expected vs observed, and an explanation of why the
output is valid — including a documented miss.

| Check | Expected | Observed |
|---|---|---|
| Smoke (100 rows) | completes; 0 missing / 0 unexpected keys | **Pass**: job 6196203, top-1 58.00% / top-5 78.00%, 5.8s eval / 14s wall, 1.86 GB peak GPU. Pipeline check only; first 100 CSV rows are not class-balanced. |
| Checkpoint keys | 0 / 0 | **0 / 0** on smoke, subset, and full |
| Stratified subset (500, seed 0, 62 classes) | >> chance (~1.61%) | **61.00%** top-1 / 77.80% top-5 (job 6197584), 40.7s eval / 58s wall. Far above chance; **not** the 79.99% claim. |
| **Full val top-1 (53,041 images)** | 79.99% ± 1.0 | **62.60%** (job 6197773). Top-5 79.77%, 935.8s eval / 994s wall, 2.05 GB peak GPU. **Misses the repository number by ~17.4 points.** |
| Paper no-TTA (context only) | 79.69 | not a pass/fail target |

Final lines of the full-val job (`logs/satmae_temporal_full_6197773.out`):

```
loaded .../finetune_fmow_temporal.pth: missing=0 unexpected=0
n=53041 top1=62.60 top5=79.77 runtime_s=935.8 peak_gpu_gb=2.05 gpu=NVIDIA L40S
wall_seconds=994
```

Smoke (`logs/satmae_temporal_smoke_6196203.out`):

```
loaded .../finetune_fmow_temporal.pth: missing=0 unexpected=0
n=100 top1=58.00 top5=78.00 runtime_s=5.8 peak_gpu_gb=1.86 gpu=NVIDIA L40S
```

Two things are checked:

1. **Clean checkpoint load.** `load_state_dict(strict=False)` reported empty
   missing and unexpected lists for `vit_large_patch16`, `global_pool=True`,
   and `--nb-classes 1000`. The weights that ran are the official ones.
2. **Accuracy vs the repository number** on the **full** validation CSV.
   62.60% is consistent with the 500-row subset (61.00%) and the smoke
   (58.00%), so the evaluator is stable — it is not a one-off crash or
   random-weight run. It is still **not** a 79.99% reproduction.

**Why 62.60% is still a valid inference result, and why it is not 79.99%.**
SatMAE never published the promised `*_rgb_crop_0.jpg` dump. SpaceNet S3
has full scenes (`*_rgb.jpg`) plus JSON boxes. This recipe reconstructs
`*_rgb_crop_0.jpg` as a **tight** `bounding_boxes[0].box` crop. SatMAE's
README says to follow the [fMoW baseline preprocess](https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py#L107),
which **pads context** around the box (15% up to 2×). A sample of our chips
has median size 204×162, and 64% are smaller than 224 on the short side.
The model was trained on padded AOIs; we scored tight boxes. That domain
shift is the leading explanation for the 17-point gap. Official eval also
used 8 GPUs and `RandomCrop` at test time; those are expected to move the
number by much less than 17 points.

The 0.30 point gap between the paper ablation (79.69) and the repository
table (79.99) is a separate upstream discrepancy and is not used as a
pass/fail target.

## Hardware and provenance

- **Verified hardware:** PACE ICE, partition `coc-gpu`, account `coc`, QOS
  `coc-ice`, 1× NVIDIA L40S (confirmed via `nvidia-smi` in every GPU job),
  `--cpus-per-task=8`, `--mem=96G` on full val (used ~21 GiB MaxRSS). An
  A100 (`--gres=gpu:a100:1`) is on the same partition but was not tested.
  Do **not** request AMD MI210 nodes (ROCm-only). CPU setup/download jobs
  use `coc-cpu`. ICE running-GPU cap: 32 GPU-hours.
- **Upstream code:** https://github.com/sustainlab-group/SatMAE at
  `0b210aceb37a14bbbd897110db5b104b3271d818` (`main`, 2025-08-10).
  License: CC BY-NC 4.0 (non-commercial use only).
- **Checkpoint:** `finetune_fmow_temporal.pth` from
  https://zenodo.org/record/7369797
  (DOI [10.5281/zenodo.7369796](https://doi.org/10.5281/zenodo.7369796)).
  SHA-256: `378c590865bb55af777f066da3d3500b4ed10b5c6b0f619b8548a541d84480a1`
  (job 6190909). Sidecar: `weights/finetune_fmow_temporal.pth.sha256`.
- **Data:** fMoW-RGB validation scenes from
  `s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/val` (~22 GB, 53,041
  `*_rgb.jpg`) plus `val_62classes.csv` from the SatMAE metadata Drive
  folder. After `satmae_temporal_crop_fmow.sbatch` (job 6195403), all 53,041
  CSV rows resolve to a reconstructed `*_rgb_crop_0.jpg`. fMoW license:
  Functional Map of the World Challenge Public License. Do not commit imagery.
- **Model config:** `vit_large_patch16`, temporal, input 224, 3 frames,
  `--nb-classes 1000` (upstream legacy; 62 real classes), `global_pool=True`,
  fp16 autocast, no TTA, batch size 8.
- **Software:** Python 3.9.18, torch 1.12.1 (CUDA 11.3), torchvision 0.13.1,
  timm 0.3.2 (patched), numpy 1.23.5. Recorded in
  `logs/smoke_6196203_versions.txt` and `logs/full_6197773_versions.txt`.
  Resolved lock + `pip freeze`: `$SATMAE_TEMPORAL_WORKDIR/env/`.
- **Seeds:** `--seed 0` in the evaluator; subset sampling uses `--seed 0`.
- **Paper:** Cong et al., NeurIPS 2022, https://arxiv.org/abs/2207.08051
- **Run dates:** 2026-10-11.
- **Resource usage:** env rebuild 17s after the failed first build (job
  6192077; first attempt 6190908 failed on `import timm`); weights 13m35s
  (job 6190909); S3 val + CSV 30m09s (job 6192078, prepare failed until
  crops existed); AOI crop 8m10s (job 6195403); smoke 22s (job 6196203);
  subset 1m21s (job 6197584); full val 17m21s (job 6197773).

## Troubleshooting

**Login-node killed / conda killed.** The login node allows about 1 CPU and 4 GB.
Submit `satmae_temporal_build_env.sbatch` to `coc-cpu`.

**Home disk full (30 GB / 96%+).** Do not install the env in `$HOME/.conda`.
The prefix is `$HOME/scratch/envs/satmae-temporal`. If `conda` still writes to
home, export `CONDA_PKGS_DIRS=$HOME/scratch/conda-pkgs` in the shell.

**`torch._six` ImportError.** `timm==0.3.2` does
`from torch._six import container_abcs`. The first env build (job 6190908)
failed here because the patch script used to `import timm` to find its
install path — that import is the broken one. The patch now rewrites files
under `site-packages/timm` without importing the package. Re-run the build
job (it will reuse the existing prefix) or
`bash scripts/satmae_temporal_patch_env.sh` inside the env.

**`nb_classes` size mismatch when loading the checkpoint.** Pass `--nb-classes 1000`.

**`transforms.Scale` missing.** Job 6195919 failed here on the pinned
torchvision 0.13.1 build. `satmae_temporal_evaluate.py` aliases `Scale` to
`Resize` (same smaller-edge resize). `satmae_temporal_patch_env.sh` also
rewrites `util/datasets.py` if you rebuild the env.

**`KeyError` on a companion `*_rgb_crop_0.jpg` during subset eval.** Job
6197109. `parse_timestamp` only indexes rows in the CSV given to the
dataset. The subset job must keep `--csv val_prepared.csv` and pass
`--eval-csv val_subset500.csv` so companion frames still have timestamps.

**`torch.arange(..., dtype=np.float64)` TypeError.** Job 6196065 loaded the
checkpoint (0 missing / 0 unexpected keys) then failed in
`util/pos_embed.py`. A naive `np.float` → `np.float64` rewrite is invalid
for `torch.arange` (it needs `torch.float64`) and also corrupted
`np.float32` → `np.float6432`. Re-run
`bash scripts/satmae_temporal_patch_env.sh` inside the env.

**gdown / Google Drive fails.** Copy `val_62classes.csv` by hand into
`$SATMAE_TEMPORAL_WORKDIR/data/` and rerun the prepare script.

**Almost every CSV path is missing.** Observed on job 6192078:
`val_62classes.csv` points at `*_rgb_crop_0.jpg` (53,041 missing) while
SpaceNet S3 only has full scenes (`*_rgb.jpg`) plus `*_rgb.json`. The
temporal loader also globs the last 15 characters (`_rgb_crop_0.jpg`), so
rewriting the CSV to `_rgb.jpg` would break companion-frame lookup. Run
`sbatch scripts/satmae_temporal_crop_fmow.sbatch` to rebuild the first AOI
chip from `bounding_boxes[0].box` (`[x, y, width, height]`) and regenerate
the prepared CSVs. These chips are reconstructed, not SatMAE's unpublished
preprocessed dump.

**Host OOM.** Raise `--mem` (64G → 96G) or lower `--num-workers`.

**Wrong GPU type / AMD MI210.** This stack is CUDA 11.3 / NVIDIA only. Request
`l40s` or `a100` explicitly.

**`mkdir: Permission denied` in a Slurm job.** Resolving the repo path from
`${BASH_SOURCE[0]}` can point at Slurm's spooled copy of the script. These
scripts use `$SLURM_SUBMIT_DIR`, so submit from the repo root.

**`MKL_INTERFACE_LAYER: unbound variable`.** Conda's MKL activate hook trips
`set -u`. Activation in the GPU preamble is wrapped in
`set +u; conda activate …; set -u`.

**`Invalid qos specification`.** Happens if the account is wrong. The
association for this recipe is `-A coc`, `-p coc-gpu` or `coc-cpu`,
`--qos=coc-ice`.

**`conda env create` fails because the prefix exists.** The build job skips
create when `$SATMAE_TEMPORAL_ENV/bin/python` is already there. Delete the
prefix to rebuild.

**Queue / 32 GPU-hour cap.** Prefer smoke + subset first. Cancel leftover GPU
jobs with `scancel <id>`.

## Cleanup

```bash
# keep the Foundry clone; remove bulky scratch artifacts
rm -rf "$HOME/scratch/satmae-temporal"
rm -rf "$HOME/scratch/envs/satmae-temporal"
rm -rf "$HOME/scratch/conda-pkgs" "$HOME/scratch/pip-cache" "$HOME/scratch/tmp"
```

Do not delete the git clone until logs and `metrics.json` excerpts have been
copied into this README or an evidence folder.

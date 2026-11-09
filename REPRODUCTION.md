# Assignment 2 — Reproducible inference (SatMAE Temporal)

This Foundry fork adds a **SatMAE Temporal ViT-Large** inference recipe for
fMoW-Temporal, verified on **PACE ICE**. The Foundry root `README.md` is
upstream project docs and is not replaced; **this file** is the assignment
entry point.

Primary runbook (setup, execution, verification, troubleshooting, cleanup):

- [registry/models/earth-science/satmae-temporal/README.md](registry/models/earth-science/satmae-temporal/README.md)
- [evidence/satmae-temporal/](evidence/satmae-temporal/) — `metrics.json`, `pip freeze`, failure notes

Related metadata:

- [registry/models/earth-science/satmae-temporal/model.yaml](registry/models/earth-science/satmae-temporal/model.yaml)
- [recipes/satmae-temporal-inference-classification.yaml](recipes/satmae-temporal-inference-classification.yaml)
- [resource-profiles/pace-ice/coc-gpu.yaml](resource-profiles/pace-ice/coc-gpu.yaml)
- [benchmarks/results/satmae-temporal/fmow-temporal-pace-ice.yaml](benchmarks/results/satmae-temporal/fmow-temporal-pace-ice.yaml)

Upstream target: **79.99% Top-1** (SatMAE README, no TTA) using
`finetune_fmow_temporal.pth`. Observed full-val Top-1 on reconstructed AOI
crops: **62.60%**. The paper ablation without TTA reports **79.69%**; that
is not treated as the same number. Foundry issue:
[#22](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/22).

## What was verified (2026-10-11, 1× L40S)

| Check | Expected | Observed |
|---|---|---|
| Smoke (100 rows, job 6196203) | 0 missing / 0 unexpected keys | 0 / 0; top-1 58.00% (pipeline only) |
| Subset (500 stratified, seed 0, job 6197584) | >> chance (1.61%) | **61.00%** / 77.80% top-5 |
| Full val (53,041 images, job 6197773) | 79.99 ± 1.0 | **62.60%** / 79.77% top-5 |

Checkpoint SHA-256:
`378c590865bb55af777f066da3d3500b4ed10b5c6b0f619b8548a541d84480a1`.
Code: sustainlab-group/SatMAE @ `0b210aceb37a14bbbd897110db5b104b3271d818`.

The 17-point miss vs 79.99% is a data-preprocess gap: SatMAE never published
the `*_rgb_crop_0.jpg` dump. This recipe reconstructs tight JSON bbox crops;
the official fMoW baseline pads context around the box. Full write-up:
[Verification](registry/models/earth-science/satmae-temporal/README.md#verification).

## Clone this fork (not `main`)

The recipe lives on **`satmae-temporal-inference`**. `main` does not have these
scripts. Clone that branch of this fork:

```bash
ssh login-ice.pace.gatech.edu
git clone -b satmae-temporal-inference \
  https://github.com/rohan5986/cybershuttle-ai4science-foundry.git \
  ~/scratch/cybershuttle-ai4science-foundry
cd ~/scratch/cybershuttle-ai4science-foundry
```

If you already have a clone, switch to the branch instead of cloning again:

```bash
git fetch origin
git checkout satmae-temporal-inference
```

## Submit jobs from the repo root on ICE

Each stage depends on the previous one. **Submit one `sbatch`, wait until
`sacct` shows `COMPLETED` with exit `0:0`, then submit the next.** Do not
queue the whole ladder at once: later jobs need the conda prefix, weights,
and cropped chips from earlier jobs.

```bash
cd ~/scratch/cybershuttle-ai4science-foundry
export SATMAE_TEMPORAL_WORKDIR="$HOME/scratch/satmae-temporal"
mkdir -p logs "$SATMAE_TEMPORAL_WORKDIR"

sinfo -p coc-gpu -o "%P %G %N"    # confirm L40S vs A100 before GPU jobs

# After each sbatch: JOBID=...; sacct -j $JOBID -n -X -o State,ExitCode
sbatch scripts/satmae_temporal_build_env.sbatch          # wait: COMPLETED
sbatch scripts/satmae_temporal_download_weights.sbatch   # wait: COMPLETED
sbatch scripts/satmae_temporal_download_data.sbatch      # wait: COMPLETED
# if prepare_csv reports 0 existing images (job 6192078):
sbatch scripts/satmae_temporal_crop_fmow.sbatch          # wait: COMPLETED
sbatch scripts/satmae_temporal_smoke_test.sbatch         # wait: COMPLETED
sbatch scripts/satmae_temporal_eval_subset.sbatch        # wait: COMPLETED
sbatch scripts/satmae_temporal_eval_full.sbatch
```

Do not run conda create, S3 sync, crop, or the evaluator on the login node.

The build job uses whichever `conda` is already on PATH (miniforge is
fine). It does **not** `module load anaconda3` on top of an existing
conda — that combination installed a torch that failed to import on a
peer ICE account. PyTorch is pip-installed as `1.12.1+cu113`.

A second full-val on a fresh clone reported **61.63%** vs **62.60%** here.
Same checkpoint and crop count; leftover noise is `RandomCrop` at eval
plus (previously) unsorted companion-frame globs. The evaluator now
sorts those globs. Neither number is the 79.99% upstream claim.

## Suggested PR title and body

**Title:** Add SatMAE Temporal (earth science) with a verified inference recipe for PACE ICE

```markdown
Adds a registry entry and PACE ICE inference recipe for SatMAE Temporal
ViT-Large on fMoW-Temporal. Related to #22.

## What's added
- `registry/models/earth-science/satmae-temporal/model.yaml`
- `registry/models/earth-science/satmae-temporal/README.md`
- `recipes/satmae-temporal-inference-classification.yaml`
- `resource-profiles/pace-ice/coc-gpu.yaml`
- `scripts/satmae_temporal_*`
- `benchmarks/results/satmae-temporal/fmow-temporal-pace-ice.yaml`
- `REPRODUCTION.md`

## Verification
Checkpoint: Zenodo 10.5281/zenodo.7369796 `finetune_fmow_temporal.pth`
(SHA-256 378c590865bb55af777f066da3d3500b4ed10b5c6b0f619b8548a541d84480a1)
Code: sustainlab-group/SatMAE @ 0b210aceb37a14bbbd897110db5b104b3271d818
Data: fMoW-RGB val from public S3 + SatMAE `val_62classes.csv` + reconstructed AOI crops

| Check | Expected | Observed |
|---|---|---|
| Smoke (100 rows, checkpoint keys) | 0 missing / 0 unexpected | 0 / 0 (job 6196203) |
| Subset (500 stratified) | >> chance (1.61%) | 61.00% (job 6197584) |
| Full val Top-1 (no TTA) | 79.99 ± 1.0 | 62.60% (job 6197773); documented crop-gap |

## Notes for reviewers
This is the temporal RGB ViT-Large variant, not multi-spectral SatMAE (#28).
`cybershuttle.enabled` is false. Nexus is untested.
```

# Provenance

This file traces every artifact the workflow depends on, with its source and version, so someone else can rebuild the same run.

## Model

FourCastNet is NVIDIA's AFNO-based global weather model (Pathak et al., 2022, https://arxiv.org/abs/2202.11214). It predicts 20 channels on a 720x1440 grid, using patch size 8, 8 AFNO blocks and an embedding width of 56.

## Code

- Repository: https://github.com/NVlabs/FourCastNet (BSD 3-Clause)
- **Commit used:** `93360c1720a9f97aabf970689f21c9fad8737788`, which every Slurm log also prints at the top
- Scripts used: `inference/inference.py` and `config/AFNO.yaml`. I patch the `afno_backbone` section with `scripts/configure_paths.py` and keep the diff in `logs/config_patch.diff`.

Reading `inference.py` shaped the workflow in four ways. It scores the `z500` field only. It uses `--vis` to run a single initial condition. It imports `numpy.core.numeric.False_`, which needs numpy below 2. It calls `torch.load` without `weights_only`, which needs torch below 2.6.

## Data bundle

The bundle comes from https://portal.nersc.gov/project/m4134/ccai_demo.tar, a gzip-compressed tar of 3,998,497,644 bytes published by the FourCastNet authors on NERSC. I recorded SHA-256 checksums for the tar and for each file I use in `docs/CHECKSUMS.sha256`, and `scripts/download_artifacts.sh` re-verifies them. The paths below sit under `ccai_demo/`.

- `model_weights/FCN_weights_v0/backbone.ckpt` (896,430,597 bytes). The bundle also holds `precip.ckpt`, which I leave unused.
- `additional/stats_v0/global_means.npy`, `global_stds.npy` and `time_means.npy`. These come from the same release as the checkpoint, so I use them as shipped and never recompute them.
- `data/FCN_ERA5_data_v0/out_of_sample/2018.h5` with dataset `fields` of shape (40, 21, 721, 1440) in float32. That is 40 six-hour steps and about 3.5 GB. The model reads the first 20 channels and the first 720 latitude rows.

The underlying data is ERA5 reanalysis from ECMWF and Copernicus. The file is the authors' preprocessed out-of-sample subset, and I downloaded no raw ERA5. The year 2018 falls outside the 1979 to 2015 training years, so the evaluation is out-of-sample. I haven't verified the calendar dates of the 40 steps.

## Software environment

I ran on PACE-ICE with `module load miniforge` and a conda environment called `fourcastnet-inference` on scratch. The spec is `environment/environment.yml` and my reasoning is in `environment/VERSION_NOTES.md`. I verified Python 3.10, torch 2.2.2+cu121 and numpy 1.26.4. The recorded run used a Tesla V100-PCIE-16GB (16,384 MiB, driver 580.178.04) on node `atl1-1-02-011-31-0`, with account `coc`, partition `ice-gpu` and QOS `coc-ice`.

## Out of scope

I left out training, `precip.ckpt`, the HuggingFace `nvidia/fourcastnet1` release with 26 channels, and the hosted APIs. Those are separate releases from the files above, so mixing them in would give a different experiment.

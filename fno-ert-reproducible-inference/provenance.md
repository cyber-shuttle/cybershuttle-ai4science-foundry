# Provenance

## Paper
Ghadjari et al. 2026, "A Fourier neural operator surrogate model for nonlinear electrical resistivity tomography",
Geophys. J. Int. 244(3), ggag022. https://doi.org/10.1093/gji/ggag022 (open access, CC BY).
Data/code availability statement points to the minimal reproduction package below. The paper releases no trained
checkpoint, so none is used; checkpoints here were trained by the author of this repo.

## Source repositories
| Artifact | Source | Version |
|---|---|---|
| Authors' code and example data | https://github.com/Hghadjari/FNO_ERT | commit `7145f02457927e3653c18b5430c67426effb7f62` (see `docs/fno_ert_commit.txt`) |
| NeuralOperator library | https://github.com/neuraloperator/neuraloperator (MIT) | commit `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`, package version 2.0.0 (see `docs/neuralop_commit.txt`) |
| MATLAB solver that generated the data (not used here) | FW2_5D, Pidlisecky and Knight 2008, Computers and Geosciences | not run |

## Input data
Two files in the authors' repo, stored with Git LFS. Fetched directly from the LFS media endpoint
(`https://media.githubusercontent.com/media/Hghadjari/FNO_ERT/main/data/<file>`) on 2026-10-09 because `git lfs pull`
did not download them.

| File | Size (bytes) | SHA-256 (matches the LFS pointer `oid`) |
|---|---|---|
| `data/evalV.mat` | 31,736,317 | `8009f1c8f9b2986705ca8a05b7c548cbea19fc18b9aa39afe6781291c3ae79b1` |
| `data/testV.mat` | 31,727,820 | `fc4411feac92ce7d06540a166215bc1dd86189b6723594b2c848be3241dce091` |

Format: MATLAB v7.3 (HDF5). Each file has three arrays `II`, `coeff`, `sol`, stored as (100, 40, 1001) in the file,
i.e. 1001 samples of a 40 x 100 grid after transposing. Per the authors' demo script: inputs are `II` and `1/coeff`;
target is `sol`. Statistics of each array are in `docs/data_inspection.txt`. The repo does not state whether the
subset uses the Kriging or the Voronoi parametrization.

Use in this work: `evalV.mat` is split 901 train / 100 validation (seeded permutation; indices saved in
`outputs/split_train_idx.npy` and `split_val_idx.npy`); `testV.mat` (1001 samples) is used only for final evaluation.

## Model
`FNO2d` class copied by import (unchanged) from `FNO_ERT/fno_demo_min.py`: 2 spectral layers, 16 x 16 retained modes,
channel width 32, input channels (II, 1/coeff) plus a 2D coordinate grid, padding 20, GELU, 2,108,001 parameters.

## Checkpoints trained here
| File | Epochs | Train job | SHA-256 |
|---|---|---|---|
| `outputs/fno_ert.pt` | 300 (best epoch 295) | 6120331 | `08780db845f3b980ddb744e4d3d2f1237d0e4cc28b940544ffb1de874f6514b2` |
| `outputs/fno_ert_1000ep.pt` | 1000 | 6284585 | `b580d5a182c766160d2b0a2b180dc222002928a2b35fbca3c53ae9b87c9d3a3c` |

## Software environment
Python 3.10.22; torch 2.14.1+cu130; numpy 2.2.6; scipy 1.15.3; h5py 3.16.0; matplotlib 3.10.9; pyyaml 6.0.3.
Full lists: `environment.yml`, `requirements.txt`, `docs/modules.txt` (loaded module: anaconda3).

## Hyperparameters and where each value comes from
| Setting | Value | Source |
|---|---|---|
| width, modes, layers | 32, 16 x 16, 2 | paper |
| batch size | 50 | paper |
| epochs | 300, then 1000 | 1000 from paper; 300 as a first run |
| learning rate, weight decay, optimizer, scheduler | 0.01, 9e-4, Adam, cosine annealing | authors' demo script (not in paper) |
| normalization | per-pixel unit Gaussian fitted on training split | authors' demo script |
| loss | per-sample relative L2 (training) | author's choice, equivalent to demo's LpLoss |
| validation fraction, seed | 10 %, 0 | author's choice |

## Compute
GT PACE ICE; Slurm account `coc`, QOS `coc-ice`; 1 GPU (A40 or L40S, recorded in each log), 4 CPUs, 16 GB.
Slurm job IDs: gpu checks 6119434, 6119564, 6119696; training 6120331 and 6284585; inference 6284517 and 6284586.

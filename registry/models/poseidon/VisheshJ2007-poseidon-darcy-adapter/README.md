# Poseidon-T (`scOT`) 1x16x16 to 4x128x128 Adapter POC

## Overview
Proof-of-concept adapter testing how a coarse 1-channel `16x16` spatial tensor (`(1, 1, 16, 16)`) can be spatially resized (`16x16 -> 128x128` via bilinear interpolation) and channel-mapped (`1 -> 4`) to match `camlab-ethz/Poseidon-T`'s expected `(B, 4, 128, 128)` input interface and time conditioning (`time=torch.tensor([1.0])`).

## Environment & Dependency Notes
- Cluster: Georgia Tech PACE ICE (`login-ice-gnr-2` CPU run for POC verification).
- Conda environment: `~/scratch/conda-envs/poseidon-poc` (Python 3.10, `torch==2.0.1`, `transformers==4.29.2`, `numpy==1.26.4`).
- Built `camlab-ethz/poseidon` (`scOT`) editable with `flit_core` and pinned `numpy<2` to avoid NumPy 2.x vs. 1.x C-ABI warnings.

## Observed Results (`adapt_darcy_poseidon.py`)
- Model load time: `157.15s`
- `zero_pad` (`1x16x16 -> 4x128x128 -> 4x128x128`): `min=-0.1570`, `max=0.4499`, `mean=0.0781`, `fwd_time=660.30s`
- `repeat` (`1x16x16 -> 4x128x128 -> 4x128x128`): `min=-0.3008`, `max=1.4636`, `mean=0.4343`, `fwd_time=621.00s`

## Limitations & Next Steps
- Spatial interpolation and zero-padding/repeating channels satisfy tensor shapes for a forward-pass check but do not replace physical multi-channel data or task-specific fine-tuning.
- Next step: test input/output layer adaptation (`ignore_mismatched_sizes=True`) and fine-tuning on GPU resources via SLURM, comparing against baseline FNO results.

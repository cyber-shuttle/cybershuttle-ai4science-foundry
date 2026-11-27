# Provenance

This file records where the model, data, and main artifacts came from. Nothing here involved retraining CHGNet.

## Model

- Model: Crystal Hamiltonian Graph Neural Network (CHGNet)
- Pretrained checkpoint ID: `0.3.0`
- Python package used: `chgnet==0.4.2`
- Checkpoint file: `chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar`
- Source repo: https://github.com/CederGroupHub/chgnet
- Checkpoint notes: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/README.md
- Paper: Deng et al. (2023), https://doi.org/10.1038/s42256-023-00716-3

The exact checkpoint SHA-256 used in each run is saved in that run's `run_metadata.json`. For my completed runs it was:

```text
d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1
```

The model weights came from the installed `chgnet==0.4.2` package. I did not download weights separately and did not tune or retrain the model.

## Data

For the benchmark I used the public Materials Project Trajectory dataset:

- Dataset: Materials Project Trajectory (MPtrj), 2022.9
- Archive: Figshare article `23713842`, version `2`
- DOI: https://doi.org/10.6084/m9.figshare.23713842.v2
- Original file: `MPtrj_2022.9_full.json`
- Figshare file ID: `41619375`
- Expected size: `12,188,168,685` bytes
- Source SHA-256: `05a675acd4f17a1dadd8a4ea6ea79fe58d3ffb28ca20277383b8f990d4787cf6`

The download manifest is saved in `docs/mptrj_source_manifest.json`.

The full archive is too large to keep in Git. I saved the deterministic subset and its checksum locally; generated input and output files are ignored by Git. These identify my original subset:

- Subset path: `inputs/mptrj_subset.json`
- Original full-file subset SHA-256: `074b888bee51ee86e1a588ce4a87da43bbd0c91e77653858952abaf9f5769a87`
- Selection: 24 structures, seed `42`, 2 to 64 atoms, one eligible frame per material ID

The full-file checksum above is for my saved file. The subset JSON includes download time and local path, so regenerating it elsewhere changes that checksum even when the selected data is identical. README Step B shows how to compare just the records. My records checksum, using the JSON serialization shown there, is `ce20a0eec71937a6cfce6d1f0ca2cfdcfa389601653860e47b426bf70124ce80`.

The subset selection is deterministic, but it is not the original CHGNet held-out test split. It should be treated as a small reproducible validation sample, not as the paper's test set.

## Label conventions

- Energy label: `energy_per_atom`, in eV/atom
- Force label: `force`, in eV/A
- Stress label: raw VASP kBar, converted with `stress_gpa = -0.1 * stress_kbar`
- Magnetic moment label: `magmom`, in mu_B. I compare absolute values and skip moment MAE for structures with missing moment labels

## Compute

- Cluster: Georgia Tech PACE ICE / AI Makerspace
- Scheduler: Slurm
- GPU inference job: `jobs/ice_gpu.sbatch`, 1 GPU, 4 CPUs, 16 GB RAM, 1 hour
- Data prep job: `jobs/prepare_mptrj.sbatch`, 4 CPUs, 16 GB RAM, 2 hours
- QoS used in my run: `coc-ice`

The actual job IDs, elapsed times, GPU model, environment, and package versions are saved in `docs/run_report.md`, `logs/`, and `results/`.

## Notes

CHGNet has its own license and the MPtrj data comes from Figshare / Materials Project related sources. The full raw dataset is not committed here. If the subset cannot be redistributed under a future policy, regenerate it from the documented source and checksum instead.

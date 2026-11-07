# Run report: CHGNet 0.3.0 on Georgia Tech ICE

Status: completed on Georgia Tech PACE ICE on October 10, 2026.

## What I ran

I ran the official pretrained CHGNet 0.3.0 model on:

- a fixed silicon crystal for the smoke test
- a deterministic 24-structure MPtrj subset for the DFT-label comparison

I did not train, fine-tune, or modify the model weights. The goal was to make the inference workflow reproducible and to keep enough evidence for someone else to check it.

## Exact execution context

- Date/time (UTC): October 10, 2026. Smoke test: `17:30:54 to 17:31:06`. Data preparation: `18:23:29 to 18:37:17`. Benchmark: `18:42:27 to 18:44:28`. These are the times in the logs.
- Job IDs: `6150476` (smoke), `6150977` (data preparation), `6151243` (benchmark).
- Course/QoS and account/partition: `coc-ice` QoS and `coc` account. `coc-gpu` partition for inference and `coc-cpu` for data preparation.
- Slurm commands used:
  ```bash
  sbatch --qos=coc-ice jobs/ice_gpu.sbatch smoke
  sbatch --qos=coc-ice jobs/prepare_mptrj.sbatch
  sbatch --qos=coc-ice jobs/ice_gpu.sbatch benchmark inputs/mptrj_subset.json
  ```
- CPU cores allocated: 4 per job.
- Host RAM requested: 16 GB per job.
- GPU type/count/VRAM: One NVIDIA A40 for each inference job (48 GB nominal GPU memory). No GPU for data preparation. The saved `nvidia_smi.txt` files have the device details.
- Walltime requested: 1 hour for each GPU inference job. 2 hours for the CPU preparation job.
- Actual Slurm elapsed time: Smoke `00:00:19`. Preparation `00:13:56`. Benchmark `00:02:07`. The benchmark's Python step ran for `00:01:52` and reported peak resident memory of `1,551,216 KB` (approximately 1.5 GB). Dataset preparation's batch step reported `12,002,928 KB` (approximately 12 GB).
- Slurm state and exit code: `COMPLETED`, `0:0` for all three jobs.
- Python/PyTorch/CUDA/CHGNet versions: Python `3.11`, PyTorch `2.5.1`, PyTorch CUDA build `12.4`, CHGNet package `0.4.2`, and pretrained checkpoint `0.3.0`. Other captured package versions: `pymatgen==2025.1.9`, `ase==3.26.0`, `numpy==1.26.4`. The full package list is in each run's `pip_freeze.txt`.
- Model 0.3.0 checkpoint filename: `chgnet_0.3.0_e29f68s314m37.pth.tar` (bundled with the CHGNet package).
- Model 0.3.0 checkpoint SHA-256: `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.

## Input and output provenance

- Smoke input: Synthetic periodic diamond silicon, cubic lattice parameter `a = 5.43 Å`, with 8 atoms. Atom 0 was displaced by `[0.03, -0.02, 0.015] Å` for the force check. The structure and predictions are in `results/slurm-6150476/smoke_predictions.json`.
- Benchmark input: Materials Project Trajectory (MPtrj) 2022.9 on Figshare, article 23713842, version 2: https://doi.org/10.6084/m9.figshare.23713842.v2. Original file: `MPtrj_2022.9_full.json`, Figshare file ID `41619375`, 12,188,168,685 bytes. Retrieval details are in `docs/mptrj_source_manifest.json`.
- Original dataset SHA-256: `05a675acd4f17a1dadd8a4ea6ea79fe58d3ffb28ca20277383b8f990d4787cf6`.
- Selection parameters: `count=24, seed=42, min_atoms=2, max_atoms=64`, with one eligible frame per material selected by deterministic SHA-256 ranking of `seed:material_id`. Of 145,923 material entries scanned, 131,483 were eligible. The final subset has 24 structures from 24 different materials.
- Subset path: `inputs/mptrj_subset.json`.
- Subset SHA-256: `074b888bee51ee86e1a588ce4a87da43bbd0c91e77653858952abaf9f5769a87`.
- Job log paths: `logs/chgnet-6150476.out`, `logs/chgnet-6150476.err`, `logs/mptrj-prep-6150977.out`, `logs/mptrj-prep-6150977.err`, `logs/chgnet-6151243.out`, `logs/chgnet-6151243.err`.
- Output paths: `results/slurm-6150476/` (smoke predictions, validation, checkpoint and environment metadata), `results/slurm-6151243/` (benchmark predictions, MAEs, checkpoint and environment metadata), and `results/prep_slurm_accounting.txt` (preparation accounting). Slurm accounting for the inference jobs is preserved within their respective output directories.

## Smoke test results

| Criterion | Expected | Observed | PASS/FAIL |
|---|---|---|---|
| All outputs finite and correct shape | Finite energy, `8 × 3` forces, `3 × 3` stress, and 8 magnetic moments | All outputs finite with correct shapes | PASS |
| Translation energy difference | < 0.001 eV/atom | 0.0 eV/atom | PASS |
| Sum-of-forces norm | < 0.1 eV/Å | 7.487740830422111 × 10⁻⁸ eV/Å | PASS |
| Finite-difference force discrepancy | < predeclared tolerance | 0.0010853409767150879 eV/Å, versus tolerance 0.16540842354297638 eV/Å | PASS |

The finite-difference check used `0.005 Å` and total energy, since CHGNet reports energy per atom. All 4/4 smoke checks passed. This checks internal consistency, not independent DFT accuracy.

Verification output: `PASS: smoke artifacts verified under results/slurm-6150476`.

## MPtrj benchmark results

| Quantity | Observed subset MAE | Number of valid components | Published full-test MAE (context only) |
|---|---:|---:|---:|
| Energy | 23.8183 meV/atom | 24 structures | 29 meV/atom |
| Force | 49.1620 meV/Å | 1,719 Cartesian components | 68 meV/Å |
| Stress | 0.20542 GPa | 216 tensor components | 0.314 GPa |
| Magnetic moment | 0.031503 μB | 369 atom values | 0.037 μB |

All 24/24 structures ran successfully, with zero failures. The MAEs are over the available reference values: energy per structure, Cartesian force components, stress tensor components, and atomic magnetic moments. Stress was converted from raw VASP kBar with `-0.1 * stress_kbar`, and magnetic moments were compared using absolute DFT values.

My interpretation: the outputs are finite and have the expected shapes, the silicon checks passed, and the subset MAEs are in the same general range as the published CHGNet 0.3.0 numbers. The subset errors are numerically below the published full-test MAEs, but that does not mean this run beats the published benchmark. These 24 structures came from the full MPtrj archive, not from a verified copy of the paper's held-out test split. Some may have been training examples.

So the benchmark shows reproducible inference and reasonable agreement with archived DFT labels. It does not reproduce the original held-out test accuracy.

Verification output: `PASS: benchmark artifacts verified under results/slurm-6151243`.

## Problems and limitations

- Attempted command that failed: Initial submission with `sbatch --qos=pace-ice jobs/ice_gpu.sbatch smoke` was rejected with `sbatch: error: Batch job submission failed: Invalid qos specification`.
- Correction: Resubmitted using the valid course QoS, `--qos=coc-ice`. This allowed the smoke, preparation, and benchmark jobs to complete.
- Nonfatal stderr warning: All successful jobs emitted `CondaError: Run 'conda init' before 'conda deactivate'`. Despite this warning, Slurm reported `COMPLETED (0:0)`, CUDA inference succeeded, and output verification passed. The warning is retained in the `.err` logs rather than hidden.
- Scientific limitations: I did not reconstruct the original held-out MPtrj partition. The 24-structure sample is small and may include training examples. I did not run new DFT and did not retrain CHGNet.

## Reproduction verification by another student

Follow `README.md` and `environment.yml` to create the environment. Then use `jobs/prepare_mptrj.sbatch` to download/select the MPtrj subset and `jobs/ice_gpu.sbatch` for smoke and benchmark inference. Use the QoS/account assigned to your ICE account.

The full 12.19 GB MPtrj archive is not committed to Git, but the URL, file ID, checksum, and retrieval procedure are documented.

After running, verify the outputs:

```bash
sha256sum -c inputs/mptrj_subset.sha256
python scripts/verify_artifacts.py --run results/slurm-<SMOKE_JOB_ID> --mode smoke
python scripts/verify_artifacts.py --run results/slurm-<BENCHMARK_JOB_ID> --mode benchmark
python -m pytest -q
```

Job IDs and elapsed times will differ on a rerun. The values above are from the saved ICE logs and generated artifacts, not estimates.

# CHGNet CPU baseline — September 27, 2026

Verified in the ICE VS Code terminal. Remote project: `/home/hice1/ppatel686/chgnet-repro`.

- Job: 5962208; 4 CPU cores, 16 GB allocated RAM, one hour.
- Python 3.10.13 via `anaconda3/2023.03`.
- CPU PyTorch 2.5.1; CHGNet checkpoint 0.3.0.
- Official source commit: `b9a6c15860237297a5a5169660e83b2e61a78e76`, package 0.4.1.
- Energy: **−7.36769152 eV/atom**.
- All energy, force, stress, and magnetic-moment comparisons passed the original numerical tolerances.
- Maximum absolute errors: energy 1.5168762e-6 eV/atom; forces 3.8147152e-6 eV/angstrom; stress 0.00040909789 GPa; moments 4.6226459e-7 mu_B.
- Model loading plus inference: **0.164 seconds**. Peak Python-process RSS: **486.0 MiB**. Timing excludes package imports and installation; RSS excludes the editor and other processes.

The earlier PyPI 0.4.0 trial passed energy/moments but failed forces/stress. Its results and log remain on ICE alongside the passing source-based run. Numerical tolerances were not changed.

Remote files: `reproduce_chgnet.py`, `README.md`, `results.json`, `run.log`, `requirements-lock.txt`, installation logs, archived release trial, and `upstream/` source checkout.

To rerun in a new ICE compute session:

```bash
module load anaconda3/2023.03
cd ~/chgnet-repro
source .venv/bin/activate
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python reproduce_chgnet.py
```

The environment and files persist after the interactive job expires. Request a new compute session to rerun. Symmetry testing and any CPU/GPU comparison remain future work. The GitHub tracking issue still awaits explicit approval.

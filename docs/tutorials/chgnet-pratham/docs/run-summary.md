# H100 execution summary

Run 6199752 executed on October 11, 2026 at 10:55 PM EDT (02:55 UTC October 12).

- Hardware: NVIDIA H100 80GB HBM3; driver 615.71.09.
- Allocation: ICE GPU partition `ice-gpu`, account `coc`, QoS `coc-ice`; one GPU, four CPU cores, 16 GB requested RAM, one-hour OnDemand session.
- Environment: Python 3.10.13; torch 2.5.1+cu121; CHGNet package 0.4.1, model checkpoint 0.3.0; NumPy 2.2.6; pymatgen 2025.10.7.
- All four smoke outputs and numerical comparisons passed, and all outputs passed three rotations, translation, and permutation validation.
- Inference: 1.317 s. Full model-load/inference/validation interval: 1.811 s. Peak RSS: 1417.6 MiB. GPU peak allocated: 175.975 MiB; reserved: 180 MiB.
- Four negative validation-unit tests passed. Inference exit status: 0.

`evidence/6199752/` contains raw output, expected values, recorded tolerances, environment locks, input/checkpoint hashes, installation and execution logs, nvidia-smi, scontrol, and sacct snapshots. The sacct snapshot was taken while the interactive session was running; MaxRSS may be unavailable until job completion. The measured Python RSS is independently recorded in results.json.

The source ID typo in the earlier summary was resolved against GitHub and the recovered historical results.json, both confirming `b9a6c15860237297a5a5169660e83b2e61a78e76`. The prior CPU 0.4.0 failure remains documented; the new CUDA run used the pinned source 0.4.1 without changing numerical tolerances.

The portal identifies this as an ICE H100 allocation; no separate allocation name saying AI Makerspace was displayed. State the actual resource faithfully rather than inventing one.

# Validation

This file explains the checks I used to decide whether the inference output was reasonable. A successful Python run is not enough by itself, since the model could still be using the wrong units, input structure, or checkpoint.

## Silicon smoke test

The smoke input is a fixed silicon diamond structure from:

```python
Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
```

I displaced atom 0 by `[0.03, -0.02, 0.015]` A so the force check is not completely trivial. The exact structure is saved in `results/slurm-6150476/smoke_predictions.json`.

| Test | Expected physical result | Threshold (predeclared) | Why it matters |
|---|---|---|---|
| Shape and finiteness | Energy scalar, forces N x 3, stress 3 x 3, magnetic moments N, all finite | Exact shapes, no NaN/Inf | Catches incorrect outputs, graph/model errors |
| Rigid translation invariance | Periodic model energy is invariant when all atoms move by the same Cartesian shift | Abs ΔE per atom < 0.001 eV/atom | Potential energy should depend on relative atomic geometry |
| Zero net force | Sum of all forces is approximately zero in a closed periodic system | Norm < 0.1 eV/Å | Translational invariance implies zero derivative along global translation |
| Force derivative | CHGNet force equals negative spatial derivative of total energy | Abs error < `0.15 + 0.05*max(|F_model|, |F_numeric|)` eV/Å | Verifies force predictions are internally consistent with energy predictions |

The finite difference uses `h = 0.005 Å` and

`F_numeric(atom 0, x) = -[E_total(x+h) - E_total(x-h)] / (2*h)`.

Important unit detail: CHGNet predicts energy per atom, but force comes from total energy. For N atoms, `E_total = N * E_predicted_per_atom`.

The smoke test checks physical consistency, not DFT accuracy. That is why I also ran the MPtrj comparison.

The thresholds were chosen before looking at the final results. If a check fails, the right thing is to keep the failed output and explain it, not silently relax the threshold.

## DFT subset benchmark

`scripts/make_subset.py` extracts 24 structures from 24 different materials. It keeps structures with 2 to 64 atoms and valid energy, force, and stress labels. Materials are ranked using `SHA256("42:" + material_id)`, so the sample is deterministic and not hand-picked.

This does not prove the structures were held out during CHGNet training.

For each structure, I compare CHGNet predictions to the archived DFT labels:

| Metric | Formula and unit | Weighting | Missing labels |
|---|---|---|---|
| Energy MAE | Mean `1000 * abs(E_CHGNet_per_atom - E_DFT_per_atom)`, in meV/atom | One observation per material/frame | Energy required |
| Force MAE | Mean `1000 * abs(F_pred - F_DFT)` per Cartesian coordinate, meV/Å | All 3N atomic force components across selected structures | Force required |
| Stress MAE | Mean `abs(S_pred_GPa - (-0.1*S_DFT_kBar))`, GPa | All 9 tensor components per structure | Stress required |
| Magnetic moment MAE | Mean `abs(m_pred - abs(m_DFT))`, mu_B | All N sites per structure with complete labels | Skip structure's moment metric if any site label missing |

The aggregate MAEs are weighted by the number of valid components. For example, force MAE is over all Cartesian force components, not one equal-weighted average per structure. `benchmark_metrics.json` records the counts.

## Published values for context

For CHGNet 0.3.0 the official checkpoint README reports test MAE: 29 meV/atom energy, 68 meV/Å force, 0.314 GPa stress, 0.037 μB magnetic moment.

My 24-structure subset is much smaller and may contain structures from the original training data. So the measured subset MAEs are only a sanity check against real DFT labels. They are not a reproduction or refutation of the published held-out test MAEs.

## How to read the saved results

Use these files together:

- `results/slurm-6150476/smoke_validation.json`
- `results/slurm-6151243/benchmark_metrics.json`
- Slurm stdout/stderr under `logs/`
- `docs/run_report.md`

The benchmark script records failures in `benchmark_metrics.json` and exits nonzero if any structure fails.

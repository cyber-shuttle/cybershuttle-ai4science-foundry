# Validation record

This file records the run I validated: Slurm job 6205007 on node `atl1-1-02-011-31-0`, a Tesla V100-PCIE-16GB, under account `coc`. The job took 95 seconds of wall time, 77 of them in inference, and peak GPU memory was 11,292 MiB (sampled twice a second in `logs/gpu_usage_6205007.csv`). The full output is in `logs/slurm-6205007.out` and `.err`. I added the CSV output to `validate.py` after this job ran. It only adds an output file and leaves the checks unchanged, so I regenerated the CSV for this job from its saved predictions.

## Smoke tests

| Check | Result |
|---|---|
| Pre-flight `scripts/smoke_test.py` (imports, numpy<2, torch<2.6, artifact files, checkpoint size, data shape) | pass (`logs/smoke_test.txt`) |
| CUDA visible on a GPU node | pass, Tesla V100-PCIE-16GB (`logs/cuda_check.txt`) |
| Artifact checksums match `docs/CHECKSUMS.sha256` | pass (`logs/download_manifest.txt`) |
| Output HDF5 found with `rmse` and `acc` | pass (`acc_coarse`, `ground_truth`, `predicted` and others are present too) |
| Output arrays finite and non-empty | pass, both have shape (1, 40, 20) |

## Substantive checks (Z500, channel 14)

| Check | Expected | Measured | Result |
|---|---|---|---|
| ACC within [-1, 1], all 20 channels | by definition | min 0.0922, max 1.0000 | pass |
| RMSE at or above 0 | by definition | min 0.0000 | pass |
| ACC at lead 0 | about 1 (initial condition) | 1.0000 | pass |
| ACC at 24 h | at least 0.6 | 0.9952 | pass |
| ACC decays and RMSE grows with lead time | steady trend | ACC 0.9952 to 0.2874, RMSE 79.61 to 1069.85 (24 h to 234 h) | pass |

Z500 skill by lead time: at 24 h, ACC is 0.9952 and RMSE is 79.61. At 48 h they are 0.9813 and 156.09, at 72 h they are 0.9437 and 271.82, and at 120 h they are 0.8252 and 478.35. The inference log shows ACC falling from 0.613 at 168 h to 0.574 at 174 h, so it crosses 0.6 at about 7 days. The last value, at 234 h (step 39), is 0.2874.

The 24 h check at 0.6 is easy for a working model to pass, since mine reaches 0.995, so the 7-day crossing tells me more. The 0.6 threshold is the long-standing convention for calling a synoptic-scale forecast useful. It comes from forecast verification practice and the FourCastNet paper doesn't define it. RMSE units are those of the Z500 field as `inference.py` scores it, and I haven't checked them independently. They sit in the same range as the m^2/s^2 values on the paper's axis.

## Comparison with the paper

The closest figure in Pathak et al. (2022) is Figure 7(a), which plots Z500 ACC and RMSE (m^2/s^2) against lead time for the unperturbed FourCastNet control. The paper averages over many forecasts across 2018 and lists about 32 to 36 initial conditions. I read the paper's values off the figure by eye, so they are approximate. My run uses one initial condition and covers 234 h, nearly the same horizon as the figure.

The two agree early. Both start at an ACC of about 1 and stay near 0.99 through 24 to 48 h (I measured 0.995 at 24 h and 0.981 at 48 h). Both curves decline smoothly, and my RMSE falls in the same range as the paper's, so the scoring looks to be on the same units.

They diverge later. My curve falls faster after about 72 h. At 96 h my ACC is 0.878 against roughly 0.93 in the paper's control, at 144 h it is 0.739 against roughly 0.83, at 192 h it is 0.443 against roughly 0.71, and at 234 h it is 0.287 against roughly 0.61. My run crosses 0.6 at about 170 h, while the paper's control stays above 0.6 across the whole plotted range. RMSE follows the same pattern, at 1070 for me and roughly 880 in the paper at 234 h.

This means my run falls short of the paper's long-lead skill. A single initial condition can differ a lot from an average over many, and ACC at 5 to 10 days varies the most, so a below-average case could explain it. I couldn't test that, because the bundled evaluation file holds only one 10-day window. A difference between this checkpoint and the one behind the paper's figure, or a difference in how ACC is computed, also remains possible. I consider agreement through about 2 days supported, and I treat the shortfall beyond about 4 days as an open discrepancy.

## Repeatability

I ran a second job (6205661) on the same node and GPU and compared it with job 6205007 using `scripts/compare_runs.py outputs/z500_skill_6205007.csv outputs/z500_skill_6205661.csv`. Over all 40 steps the largest absolute difference was 0.000e+00 for both ACC and RMSE, so the pipeline gives identical results on this hardware. Both jobs used the same node and GPU, so this says nothing about drift between different GPU types or cuFFT versions.

## Verdict

The pipeline ran from checksummed artifacts to scored output twice with identical results, and the output looks physically sensible. It starts from the correct initial state, skill decays smoothly, and the curve agrees with the paper's Figure 7(a) through about 2 days. Beyond about 4 days my skill is clearly below the paper's average curve, and this run can't tell me whether that comes from single-initialization variance or a real discrepancy. It supports the claim that the workflow works and produces a sensible forecast. It leaves the paper's long-lead skill unconfirmed. To settle it, I would evaluate many independent initial conditions (that needs the full 2018 file, since the demo bundle holds a 40-step window), confirm the calendar dates of these 40 steps, and repeat the run on a different GPU type.

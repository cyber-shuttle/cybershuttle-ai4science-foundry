# Reproducible inference: FNO surrogate for nonlinear ERT (GT PACE ICE)

Reproduces inference with the Fourier Neural Operator (FNO) forward-model surrogate for electrical resistivity
tomography (ERT) from Ghadjari et al. 2026 (Geophys. J. Int. 244(3), ggag022, https://doi.org/10.1093/gji/ggag022),
on the Georgia Tech PACE ICE cluster. The authors release no trained weights, so a small checkpoint is trained here
from their example data, then used for inference and validated against their separate test file.

**Headline result (read this first):** the workflow runs end to end and the smoke test passes, but accuracy does
**not** reproduce the paper. Mean relative L2 error on the 1001-sample test file is **20.57 %** (1000-epoch run) and
**21.69 %** (300-epoch run), versus the paper's reported mean error of below 0.3 %. The model is about 4x better than
a trivial mean-field baseline (94.57 %), so it learned something real, but it is not accurate. See "Results" and
"Why the gap" below.

## What is in this repo

| Path | Purpose |
|---|---|
| `scripts/common.py` | data loading, normalizer, model builder, metrics (imports the authors' `FNO2d`) |
| `scripts/inspect_data.py` | prints shapes/ranges of the `.mat` files |
| `scripts/train.py` | trains the checkpoint (90/10 train/val split of `evalV.mat`) |
| `scripts/smoke_test.py` | one-sample inference check (shape, finite, error < 1.0) |
| `scripts/infer.py` | inference on all of `testV.mat`, timing, checkpoint hash |
| `scripts/validate.py` | error metrics, baseline comparison, plots |
| `configs/fno_ert.yaml`, `configs/fno_ert_1000ep.yaml` | hyperparameters (300 and 1000 epochs) |
| `slurm/*.slurm` | job scripts (`gpu_check`, `train`, `infer`, and the 1000-epoch variants) |
| `environment.yml`, `requirements.txt` | exact package versions |
| `logs/` | stdout/stderr of every Slurm job, including the failed/awkward ones |
| `outputs/run1_300ep/`, `outputs/run2_1000ep/` | metrics, summaries, histories, figures for each run |
| `outputs/fno_ert.pt`, `outputs/fno_ert_1000ep.pt` | the two checkpoints (about 8 MB each) |
| `docs/provenance.md` | sources, commits, checksums, assumptions |
| `docs/troubleshooting.md` | every problem hit and how it was fixed |

## Compute used

GT PACE ICE (Slurm). Account `coc`, QOS `coc-ice` (partition `coc-gpu` is selected automatically). Each job: 1 GPU,
4 CPUs, 16 GB RAM. Training and inference jobs landed on whichever GPU was free (L40S or A40); the GPU used for each
run is printed at the top of its log. Training is under 10 minutes and inference under 3 minutes per job.

## Reproduce from scratch

All paths below are specific to the author's account (`/storage/ice1/0/5/ityagi8`). Replace them with your own scratch
directory, and edit the same paths in `configs/*.yaml` and `slurm/*.slurm`.

1. **Access.** Connect to the GT VPN if off campus, then `ssh <gtusername>@login-ice.pace.gatech.edu`.
   Find your Slurm account and QOS with `sacctmgr show assoc user=$USER format=account%30,partition,qos%40`
   and put them in the `#SBATCH -A` / `-q` lines of the job scripts.
2. **Clone this repo** into your scratch directory:
   `git clone https://github.com/ish067/fno-ert-reproducible-inference.git`
3. **Environment.**
   ```
   module load anaconda3
   conda create -p <scratch>/envs/fno-ert python=3.10 -y
   conda activate <scratch>/envs/fno-ert
   pip install torch numpy scipy matplotlib h5py pyyaml
   ```
   For exact versions see `environment.yml` / `requirements.txt` (Python 3.10.22, torch 2.14.1+cu130, numpy 2.2.6,
   scipy 1.15.3, h5py 3.16.0, matplotlib 3.10.9, pyyaml 6.0.3). The nodes' driver reports CUDA 13.4, so the default
   pip wheel works; no CUDA module is loaded. Job scripts call the environment's Python by absolute path instead of
   `conda activate` (see troubleshooting).
4. **Authors' code and data.**
   ```
   git clone https://github.com/Hghadjari/FNO_ERT.git      # commit 7145f02457927e3653c18b5430c67426effb7f62
   cd FNO_ERT/data
   curl -L -o evalV.mat https://media.githubusercontent.com/media/Hghadjari/FNO_ERT/main/data/evalV.mat
   curl -L -o testV.mat https://media.githubusercontent.com/media/Hghadjari/FNO_ERT/main/data/testV.mat
   sha256sum evalV.mat testV.mat
   ```
   The files in the git clone are Git LFS pointer stubs (133 bytes); `git lfs pull` did not fetch them here, hence the
   direct `curl`. Checksums must match `docs/data_checksums.txt`
   (`8009f1c8...79b1` for evalV, `fc4411fe...e091` for testV).
5. **Inspect the data (optional):** `<env>/bin/python scripts/inspect_data.py <evalV> <testV>`
   (output saved in `docs/data_inspection.txt`).
6. **GPU check (optional):** `sbatch slurm/gpu_check.slurm`; expect `2.14.1+cu130 True 2.0.0` in the log.
7. **Train:** `sbatch slurm/train.slurm` (300 epochs) or `sbatch slurm/train_1000ep.slurm` (1000 epochs).
   Writes the checkpoint and `outputs/train_summary.json`.
8. **Inference + validation:** `sbatch slurm/infer.slurm` (or `infer_1000ep.slurm`). One job runs the smoke test,
   inference, and validation in order. Wait until `squeue -u $USER` is empty before reading the log; the
   job runs in about 2.5 minutes.
   To run the 1000-epoch pair with automatic ordering:
   `J=$(sbatch --parsable slurm/train_1000ep.slurm); sbatch --dependency=afterok:$J slurm/infer_1000ep.slurm`
   Note: `train.py`, `infer.py`, `validate.py` write fixed filenames in `outputs/` (summaries, metrics, plots), so
   copy them aside between runs (this is what `outputs/run1_300ep/` and `run2_1000ep/` contain).

## What counts as a successful run

- Smoke test prints four `PASS` lines (shape `(1, 40, 100, 1)`, finite, matches ground truth shape, sample error < 1.0).
- Inference prints a `SUMMARY` with the checkpoint's sha256 and timings.
- Validation prints the expected-vs-observed table and writes `outputs/metrics.json`, `example_predictions.png`,
  `error_hist.png`. The pass criterion fixed before running was: model error below 0.5x the trivial mean-field baseline.

## Results (test file `testV.mat`, 1001 samples)

| Run | Checkpoint SHA-256 | Best val rel. L2 | Test mean rel. L2 | std | median / p95 / max | Baseline |
|---|---|---|---|---|---|---|
| 300 epochs (train job 6120331, infer job 6284517) | `08780db8...6514b2` | 10.03 % | **21.69 %** | 21.64 % | 16.76 / 53.45 / 407.69 % | 94.57 % |
| 1000 epochs (train 6284585, infer 6284586) | `b580d5a1...9d3a3c` | about 7.8 % | **20.57 %** | 25.77 % | 16.10 / 43.54 / 709.93 % | 94.57 % |

Paper target: mean relative error below 0.3 %, standard deviation below 5 % (paper's test set and metric
definition may differ, so the comparison is indicative only). **Paper target met: False.**

Inference speed (batched, 1001 samples): about 0.2 ms per sample; single-sample forward pass about 1.5 ms (NVIDIA A40,
warmed up, after the first call). The very first forward pass in the smoke test took 22.6 s because of one-time CUDA
and FFT initialization; that figure is not a speed measurement. The paper cites about 0.01 s per forward evaluation;
hardware and what is timed differ, so treat the speed comparison as qualitative.

## Why the gap (hypotheses, not verified)

- **Training data size:** 901 training samples here (a 90 % split of the 1001-sample `evalV.mat`), versus a larger
  training set in the paper. The paper's training set is not part of the released example data.
- **Distribution shift:** validation error (about 7.8 %) is far below test error (about 20.6 %), and the worst test
  sample reaches 700 % relative error, so some test samples are much harder or differently distributed than `evalV`.
  Relative error also explodes when a sample's true field has small norm.
- **Hyperparameters not given in the paper** were taken from the authors' demo script (lr 0.01, weight decay 9e-4,
  Adam with cosine schedule, padding 20). Width 32 and 1000 epochs follow the paper.
- 1000 epochs improved validation error (10.0 % to about 7.8 %) but changed the test mean only slightly
  (21.7 % to 20.6 %), which points to data limits rather than training length.

## Deviations from the authors' demo (`fno_demo_min.py`)

- Width 32 (demo uses 18), 300 and 1000 epochs (demo 100).
- The demo selects its checkpoint using the test set; here 10 % of `evalV.mat` is held out for validation and
  `testV.mat` is used only for final evaluation.
- The model class `FNO2d` is imported unchanged from the authors' file. The `neuraloperator` library
  (commit `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`, v2.0.0) is installed and verified by `gpu_check.slurm`, but is not
  used to build the model.

## Not done

Stage B (fine-tuning on a modified dataset) and MCMC inversion were out of scope for the time available.

## Clean up

```
scancel -u $USER                                  # cancel any queued jobs
conda env remove -p <scratch>/envs/fno-ert        # remove the environment
rm -rf <scratch>/FNO_ERT <scratch>/neuraloperator # remove data and cloned code
```

## Troubleshooting

See `docs/troubleshooting.md` (LFS pointer files, conda inside Slurm, git lock file, token authentication,
pasted-code corruption, first-call GPU latency, dropped SSH sessions).

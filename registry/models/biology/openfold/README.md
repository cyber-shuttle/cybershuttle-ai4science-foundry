# OpenFold — Reproducible Inference on Georgia Tech ICE

This directory is a complete, re-runnable reproduction of OpenFold's headline claim:
**OpenFold's openly trained weights match AlphaFold2's accuracy** (Ahdritz et al., *Nature Methods* 2024, Fig. 1).
OpenFold and AlphaFold2 (AF2) weights run through the same OpenFold v2.2.0 code, on identical
inputs, for CAMEO targets from OpenFold's own validation window. The predictions are then scored against the
experimental structures with OpenFold's validation metric (lDDT-Cα).

**Result: all three success criteria pass.** On 109 CAMEO chains, OpenFold `finetuning_ptm_2`
scores a mean lDDT-Cα of **0.9096**; the published value is **0.9097**. On identical inputs, OpenFold and AF2 are
statistically indistinguishable: the paired mean difference is −0.002 with templates and +0.001 without. Upstream's
bundled example is reproduced to 0.05 Å, and repeated runs are byte-identical across environment builds and GPU models.

- [What was run](#what-was-run)
- [Reproduce it](#reproduce-it) · [Where results are written](#where-results-are-written)
- [Validation results](#validation-results)
- [Provenance](#provenance)
- [Compute resources](#compute-resources)
- [Troubleshooting and deviations](#troubleshooting-and-deviations)
- [Limitations](#limitations)
- [Licenses and citation](#licenses-and-citation)

## What was run

| Step | Script | Where | What it does |
|---|---|---|---|
| 1. Setup | [scripts/setup.sh](scripts/setup.sh) | Slurm CPU job (`ice-cpu`) | Installs Miniforge, clones OpenFold at the pinned commit, creates the conda env from [environment.yml](environment.yml), builds OpenFold's CUDA kernel, downloads and checksums the weights, then runs [scripts/prepare_data.py](scripts/prepare_data.py) to stage inputs |
| 2. Smoke test | [scripts/run_inference.sh](scripts/run_inference.sh) `smoke` | Slurm GPU array (`ice-gpu`, H100/H200) | Upstream's bundled 6KWC example with all four weight sets |
| 3. Inference | [scripts/run_inference.sh](scripts/run_inference.sh) `cameo` | Slurm GPU array (`ice-gpu`, H100/H200) | 109 CAMEO chains × 4 weight sets, one array task per weight set |
| 4. Evaluation | [scripts/evaluate.py](scripts/evaluate.py) | login node (≈2 min CPU) | lDDT-Cα, GDT-TS and Cα-RMSD against the experimental structures, plus the success criteria |

[scripts/config.sh](scripts/config.sh) holds every path and version shared by the scripts. Large
artifacts (env, weights, inputs, predictions; ≈30 GB) live under `$OF_ROOT` (default
`~/scratch/openfold-repro`), never in git.

**Weight sets.** Each OpenFold checkpoint is paired with the AF2 model that uses the same configuration preset:

| Array task | Name | `--config_preset` | Weights | Templates |
|---|---|---|---|---|
| 0 | `openfold_templates` | `model_1_ptm` | OpenFold `finetuning_ptm_2.pt` | yes |
| 1 | `alphafold2_templates` | `model_1_ptm` | AF2 `params_model_1_ptm.npz` | yes |
| 2 | `openfold_no_templates` | `model_3_ptm` | OpenFold `finetuning_no_templ_ptm_1.pt` | no |
| 3 | `alphafold2_no_templates` | `model_3_ptm` | AF2 `params_model_3_ptm.npz` | no |

All runs use `--data_random_seed 42`, `--skip_relaxation`, precomputed alignments, and a template
cutoff of `--max_template_date 2021-10-10` (the cutoff in OpenFold's training commands). Every evaluation target was
released after that date, so no target can serve as its own template.

**Evaluation set.** The targets are the CAMEO 3D targets from OpenFold's validation window: 13 weekly releases,
2021-10-23 to 2022-01-15 (the paper's "three-month period ending on January 16, 2022"), ≤700
residues. That gives 181 chains, matching the docs' "approximately 180". Of these, 109 have precomputed OpenProteinSet
alignments. They are every chain from the first 8 releases (2021-10-23 to 2021-12-11); the 72 chains from
the last 5 releases came out after OpenProteinSet's PDB snapshot. Each candidate's status, input
checksums and selected templates are listed in [results/data_manifest.tsv](results/data_manifest.tsv).

**Success criteria (set in advance in the model-selection report)**

1. Mean lDDT-Cα within ±0.02 of the published value for each OpenFold checkpoint.
2. Paired OpenFold − AF2 mean lDDT-Cα difference of at most 0.01 in absolute value, with a 95% bootstrap CI.
3. Smoke test within 0.5 Å Cα-RMSD of upstream's bundled 6KWC prediction.

## Reproduce it

**Prerequisites.** An ICE account. All commands run from the login node, which has outbound
internet access; GPUs are reached only through Slurm. You need about 40 GB free in `~/scratch` (300 GB quota). No
sequence databases are needed. To run elsewhere, set `OF_ROOT` and adjust the `#SBATCH` partition
and constraint lines.

```bash
# Before PR #44 is merged: the contributor's fork, branch add-openfold.
# After the merge: clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git and skip the checkout.
git clone https://github.com/Rosette2237/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry
git checkout add-openfold
cd registry/models/biology/openfold   # every script runs from here

setup=$(sbatch --parsable scripts/setup.sh)                                      # ~20 min, CPU
smoke=$(sbatch --parsable --dependency=afterok:$setup scripts/run_inference.sh smoke)   # ~2 min, 4 GPUs
cameo=$(sbatch --parsable --dependency=afterok:$smoke scripts/run_inference.sh cameo)   # ~25 min, 4 GPUs
squeue -u $USER                                                                  # wait until empty

source scripts/config.sh && activate_env
python scripts/evaluate.py | tee results/logs/evaluate.log                       # writes metrics.csv + summary.json
```

Each step can also be submitted on its own. `setup.sh` skips any stage whose output already exists, so a failed
job can simply be resubmitted. To use another GPU type, override the constraint on the command line, e.g.
`sbatch -C gpu-a100 scripts/run_inference.sh cameo` (the kernel is built for sm_52/61/70/80/90).

**Verify your run** (in the same shell, after `source scripts/config.sh`)

```bash
# 1. Weights: setup.sh checks the downloads against results/weights.sha256 ("OK" lines in its log).
# 2. Inputs: results/data_manifest.tsv is regenerated; it should be identical to the committed one.
git diff --stat results/data_manifest.tsv
# 3. Smoke predictions: byte-identical to the committed ones on H100/H200. Other GPUs (e.g. V100, no TF32)
#    differ slightly by design; compare their scores in results/metrics.csv instead.
for c in openfold_templates alphafold2_templates openfold_no_templates alphafold2_no_templates; do
  cmp $OF_RUNS/smoke/$c/predictions/6KWC_1_*_unrelaxed.pdb results/predictions/smoke/$c.pdb && echo "$c identical"
done
# 4. Scores: compare results/summary.json with the committed version (git diff).
```

### Where results are written

Outputs go to two places: large files under `$OF_ROOT`, and small, versioned files in this directory.

- **`$OF_ROOT` on ICE** defaults to `~/scratch/openfold-repro`, i.e. `/storage/ice1/<…>/<user>/openfold-repro`.
- **`$OF_ROOT` on other HPC systems:** run `export OF_ROOT=/path/on/your/scratch/openfold-repro` before `sbatch`. Slurm passes
  the variable to the jobs. Also adjust the `#SBATCH -p`/`-C` lines for your partitions.

| What | Path |
|---|---|
| Predicted structures (pLDDT in the B-factor column) | `$OF_ROOT/runs/<set>/<weight set>/predictions/<tag>_<preset>_unrelaxed.pdb` |
| Per-target GPU time | `$OF_ROOT/runs/<set>/<weight set>/timings.json` |
| GPU memory/utilization trace | `$OF_ROOT/runs/<set>/<weight set>/gpu_usage.csv` |
| Staged inputs | `$OF_ROOT/data/{fasta,alignments,mmcif}/` |
| Per-prediction scores | `results/metrics.csv` |
| Means, paired differences, criteria | `results/summary.json` |
| Target list, input checksums, templates | `results/data_manifest.tsv` |
| Weight checksums | `results/weights.sha256` |
| Slurm logs | `results/logs/openfold-setup-<jobid>.out`, `results/logs/openfold-infer-<jobid>_<task>.out` |
| Evaluation log | `results/logs/evaluate.log` |

`<set>` is `smoke` or `cameo`; `<weight set>` is one of the four names in the table above. `results/` paths are
relative to `registry/models/biology/openfold/`.

## Validation results

All numbers below come from [results/summary.json](results/summary.json). Per-target scores for all
436 CAMEO predictions and the 4 smoke predictions are in [results/metrics.csv](results/metrics.csv), and the run log is in
[results/logs/evaluate.log](results/logs/evaluate.log).

**CAMEO, 109 chains (34–678 residues), each weight set on identical inputs**

| Weight set | Mean lDDT-Cα | Published | Δ | Mean GDT-TS | Median Cα-RMSD | Mean pLDDT | r(pLDDT, lDDT-Cα) |
|---|---|---|---|---|---|---|---|
| OpenFold, templates (`finetuning_ptm_2`) | **0.9096** | 0.9097 | −0.0001 | 0.821 | 1.44 Å | 89.4 | 0.88 |
| AF2, templates (`model_1_ptm`) | 0.9119 | — | — | 0.818 | 1.50 Å | 88.9 | 0.81 |
| OpenFold, no templates (`finetuning_no_templ_ptm_1`) | **0.9093** | 0.9025 | +0.0068 | 0.814 | 1.56 Å | 89.7 | 0.82 |
| AF2, no templates (`model_3_ptm`) | 0.9086 | — | — | 0.805 | 1.59 Å | 89.6 | 0.83 |

**Success criteria**

| Criterion | Measured | Pass |
|---|---|---|
| 1. Mean lDDT-Cα within ±0.02 of published | Δ = −0.0001 (templates), +0.0068 (no templates) | ✅ ✅ |
| 2. \|paired OpenFold − AF2\| ≤ 0.01 | −0.0024, 95% CI [−0.0114, +0.0038] (templates); +0.0007, CI [−0.0036, +0.0058] (no templates) | ✅ ✅ |
| 3. 6KWC within 0.5 Å Cα-RMSD of the bundled prediction | 0.049 Å (OpenFold weights), 0.373 Å (AF2 weights) | ✅ |

**Interpretation**

- **Parity with AF2.** Per-target lDDT-Cα of OpenFold and AF2 correlate at r = 0.92 (templates) and 0.97 (no
  templates). OpenFold scores at least as well as AF2 on 43% and 42% of targets. The small negative template-mode mean comes
  mostly from one hard target, `7bcj_A`: it has 19 MSA sequences and no template passes the cutoff, all four weight sets score
  only 0.27–0.66 on it, and every one of them flags it with low pLDDT (33–59). Without `7bcj_A` the difference is +0.0012. The
  lower CI bound (−0.0114) extends just past −0.01, so a deficit slightly larger than 0.01 cannot be ruled out at
  95% confidence with 109 chains.
- **Published values.** The templated checkpoint reproduces its published validation score to the fourth decimal.
  The template-free checkpoint scores 0.007 above its published value, inside the ±0.02 criterion. That is expected,
  because only 109 of the authors' ≈180 chains could be used, and a different subset shifts a mean by this much.
- **Confidence is informative.** Mean pLDDT tracks the true per-target lDDT-Cα (Pearson r 0.81–0.88).
- **Smoke test.** All four 6KWC predictions score lDDT-Cα 0.963–0.983 against the experimental structure (PDB 6KWC). The
  bundled reference `6KWC_1_model_1_ptm_unrelaxed.pdb` matches the *OpenFold* checkpoint to 0.049 Å (pLDDT 97.44 vs 97.47),
  even though `examples/monomer/inference.sh` defaults to AF2 weights. It was evidently produced with OpenFold weights;
  AF2 weights still land within 0.37 Å.

## Provenance

| Artifact | Source | Version / identifier | Retrieval |
|---|---|---|---|
| OpenFold code | <https://github.com/aqlaboratory/openfold> | tag `v2.2.0`, commit `e938c184a291bf053af3b14c1e3e8bb29aee57e2` | `git clone` + `checkout` in `setup.sh` |
| Python environment | [environment.yml](environment.yml) | full pin (225 conda packages with build strings + 10 pip packages), Python 3.10.22, PyTorch 2.5.1, CUDA 12.4 | `conda env create` with Miniforge `26.5.3-0` |
| Stereochemistry table | OpenStructure `stereo_chemical_props.txt` (LGPL-3.0), `git.scicore.unibas.ch/schwede/openstructure` at commit `7102c636`; fallback: [Software Heritage blob](https://archive.softwareheritage.org/api/1/content/sha1_git:25262efd7689fc76f4d6ea1a7c75e342e840ea1d/raw/) | sha256 `24510899eeb49167…` (asserted) | `setup.sh` (60 s timeout); read only by relaxation/training |
| OpenFold weights | `s3://openfold/openfold_params/` (Registry of Open Data on AWS) | release `v06_22`: `finetuning_ptm_2.pt`, `finetuning_no_templ_ptm_1.pt` | upstream `scripts/download_openfold_params.sh`; sha256 in [results/weights.sha256](results/weights.sha256) |
| AF2 weights | `https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar` | `params_model_1_ptm.npz`, `params_model_3_ptm.npz` | upstream `scripts/download_alphafold_params.sh`; sha256 in [results/weights.sha256](results/weights.sha256) |
| CAMEO target list | CAMEO 3D `1-year` target list ending 2022-01-15, [Wayback snapshot 2023-08-07](https://web.archive.org/web/20230807161458id_/https://cameo3d.org/modeling/targets/1-year/ajax/?to_date=2022-01-15) | sha256 `b7b680a09536de3f…` (asserted) | `prepare_data.py` |
| Alignments + template hits | OpenProteinSet, `s3://openfold/pdb/<chain>/{a3m,hhr}` | public bucket (accessed 2026-10-12 and 2026-10-26, identical); per-chain sha256 in the manifest | `prepare_data.py` (HTTPS, no AWS account) |
| Duplicate-chain map | `s3://openfold/duplicate_pdb_chains.txt` | sha256 `73cf5d547cfcb291…` (logged) | `prepare_data.py` |
| Template release dates | `s3://openfold/data_caches/mmcif_cache.json` | sha256 `ab8a3ad21a8cfcde…` (logged) | `prepare_data.py`; passed as `--release_dates_path` |
| Obsolete-PDB map | wwPDB `obsolete.dat`, [Wayback snapshot 2022-06-21](https://web.archive.org/web/20220621163601id_/https://files.rcsb.org/pub/pdb/data/status/obsolete.dat) | sha256 `8f14c08dd4d77fad…` (asserted) | `prepare_data.py`; passed as `--obsolete_pdbs_path` |
| Experimental and template structures | RCSB PDB mmCIF, `https://files.rcsb.org/download/<ID>.cif` | current entry versions (accessed 2026-10-12 and 2026-10-26, identical); target mmCIF sha256 in the manifest | `prepare_data.py` (templates fetched on demand by OpenFold's own featurizer) |
| Smoke-test input and reference | `openfold/examples/monomer/` (6KWC chain A, 191 aa) | same commit | in the OpenFold clone |

## Compute resources

Everything ran on Georgia Tech ICE (Slurm, QOS `coc-ice`); see
[resource-profiles/ice/ice-h100-h200.yaml](../../../../resource-profiles/ice/ice-h100-h200.yaml). The numbers below are from
the canonical run (logs in [results/logs/](results/logs/); `sacct -j <id>` reproduces the accounting).

| Step | Job | Node / GPU | Requested | Elapsed | Notes |
|---|---|---|---|---|---|
| Setup | 6601181 | `ice-cpu`, atl1-1-01-002-36-0 | 8 CPU, 64 GB, 4 h | 18 min 25 s | Clean build from an empty `$OF_ROOT`; MaxRSS 33 GiB |
| Smoke | 6601182_[0-3] | `ice-gpu`: H200 141 GB (2 nodes) | 1 GPU, 8 CPU, 64 GB each | 4:29, 2:59, 0:25, 0:22 | The first task on each node also loads the env from Lustre |
| CAMEO | 6601183_[0-3] | `ice-gpu`: H200 141 GB (tasks 0, 1, 3 on one node; task 2 on another) | 1 GPU, 8 CPU, 64 GB each | 23:27, 22:37, 59:56, 23:00 | Task 2 was slow; see below |
| Evaluation | — | login node, 1 CPU | — | ≈2 min | |

- **GPU time:** 2.3 GPU-hours allocated in total, of which 1.6 h is model forward passes (`timings.json`). Per-target
  GPU time on an H200 (OpenFold, templates) has a median of 2.4 s below 200 residues, 6.7 s for 200–400, and 17.2 s for
  400–700; the maximum is 36 s at 678 residues. That is about 10% faster than on an H100 in the first run. CPU featurization (MSA parsing,
  kalign template realignment) adds a few seconds per target.
- **Slow task:** in both full runs, the `openfold_no_templates` task (both times on an H200) ran about 3.5× slower than
  the other three. In the first run, nvidia-smi caught that GPU in thermal slowdown (86–87 °C, SM clock down to 345 MHz;
  [evidence](results/logs/troubleshooting/gpu-thermal-throttle-6201044_2.txt)); the second run's node was not instrumented.
  Both `model_3_ptm` weight sets run the same compute graph, and this one was not slow in the smoke tests, so a hardware
  cause is likely but unproven. The predictions are byte-identical either way.
- **Memory:** every target (≤ 678 residues) fits on an 80 GB H100 (first run) with OpenFold's default chunking. nvidia-smi
  reports the card as nearly full (80 GB on H100, 140 GB on H200) only because PyTorch's caching allocator grows into free
  memory, so that figure is not a requirement. Host RAM peaked at about 10 GiB per GPU task.
- **Storage:** about 30 GB on scratch: env 13 GB, Miniforge and package cache 6.8 GB, weights 8.7 GB (the full OpenFold and
  AF2 parameter sets, of which 4 files are used), inputs 1.7 GB (609 mmCIFs, 110 alignment sets), predictions 70 MB. About
  16 GB is downloaded (conda packages ≈6 GB, weights ≈9 GB, inputs ≈1.7 GB).

## Troubleshooting and deviations

Each problem below was hit during development or peer review, diagnosed, and fixed in the scripts. Logs of
failed and earlier runs are kept in [results/logs/troubleshooting/](results/logs/troubleshooting/).

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | The CAMEO endpoint used by upstream's `scripts/download_cameo.py` returns HTTP 503 | CAMEO moved its 3D targets API to `archive.cameo3d.org`, which is offline; the new site has no equivalent endpoint | Use the Wayback Machine's archived copy of the 1-year target list (pinned by sha256) and select OpenFold's 3-month window from it |
| 2 | Kernel build: `x86_64-conda-linux-gnu-c++: fatal error: cannot execute 'cc1plus'` (job 6200320) | Upstream `environment.yml` installs only `gcc`; the C++ compiler is in `gxx` | Added `gxx=12.4` to the env |
| 3 | Same error with `gxx` installed (job 6200530) | PACE's default `gcc/12.3.0` module exports `GCC_ROOT`, which conda-forge's GCC uses to find `cc1plus`, so it searched the system GCC tree | `activate_env` unsets `GCC_ROOT`. `setup.sh` checks `import attn_core_inplace_cuda` after the build, since upstream's installer exited 0 on failure |
| 4 | `flash-attn` in upstream `environment.yml` | Sdist-only package that imports torch at build time, so it fails in pip's isolated build; it is also unused for inference | Removed |
| 5 | `ValueError: Could not find CIFs in …/templates` (job 6200624) | OpenFold's template featurizer refuses an empty mmCIF directory | Experimental structures and templates share one `mmcif/` directory, as in upstream's training layout. The date cutoff (and an assertion) keeps targets from being their own templates |
| 6 | `RecursionError` in `_parse_obsolete` (job 6200665) | Today's wwPDB `obsolete.dat` contains a self-referencing entry (`8UJJ → 8UJJ`, 2025-12-24) that OpenFold v2.2.0 follows forever | Use a 2022-06-21 snapshot, which matches the date of OpenProteinSet (pinned by sha256) |
| 7 | 72 of 181 window chains have no alignment | OpenProteinSet's PDB snapshot ends mid-December 2021 | Evaluate on the 109 covered chains (the complete first 8 weekly releases), with the exclusions listed in the manifest |
| 8 | Setup MaxRSS reached its 32 GB request (job 6201042) | Slurm counts page cache from the multi-GB downloads | `setup.sh` now requests 64 GB; the clean rerun (6601181) peaked at 33 GiB |
| 9 | Setup hangs downloading `stereo_chemical_props.txt` (PR #44 peer review) | Upstream's `install_third_party_dependencies.sh` uses `wget -N` with no timeout (defaults: 20 tries × 900 s) against one third-party host | `setup.sh` skips the installer. It fetches the file with a 60 s timeout and a Software Heritage fallback, checks its sha256, and runs `python setup.py install`. The installer's other steps are test/training-only; `KMP_AFFINITY=none` is kept in `config.sh` |

`prepare_data.py` runs OpenFold's own `DataPipeline.process_fasta` on every target before any GPU
time is spent, and downloads each template mmCIF the featurizer asks for. The GPU jobs therefore read exactly the
files that were staged, and the templates OpenFold selected are recorded in the manifest.

**Determinism.** Every prediction was reproduced byte-for-byte across independent clean builds:
- the 4 smoke predictions three times: an environment solved from upstream's loose spec (job 6200939), then the pinned
  [environment.yml](environment.yml) (jobs 6201043 and 6601182);
- all 436 CAMEO predictions twice, two weeks apart (jobs 6201044 and 6601183).

The runs used different nodes and both H100 and H200 GPUs. The re-staged input manifest was also identical.

## Limitations

- **Subset of the validation set.** 109 of ≈181 chains have public precomputed alignments. The published means
  were computed on the full set, so they are compared on a subset. The paired OpenFold-vs-AF2 comparison is unaffected,
  because both weight sets see identical inputs.
- **Possible train/validation overlap.** The paper does not say whether these CAMEO chains were excluded from
  OpenFold's training set, and OpenProteinSet (its training data) does contain alignments for them. AF2's training
  data ends in April 2018, so it cannot have seen them. Overlap would favor OpenFold, but OpenFold is not better than AF2 here
  (paired difference ≈ 0), so there is no sign that it inflates the result.
- **Protocol scope.** One seed (42), no ensembling, no Amber relaxation, monomers only. Accuracy is measured on Cα
  (lDDT-Cα, GDT-TS, RMSD); side-chain and stereochemistry quality are not assessed.
- **Live inputs.** RCSB mmCIFs and the OpenProteinSet bucket are fetched as they are today. The manifest records sha256
  checksums so any later drift is detectable. Two inputs come from Wayback snapshots, which pins them but depends on
  archive.org.
- **Hardware.** Validated on H100 and H200. In the PR #44 peer review, the smoke test and the five shortest CAMEO
  chains also reproduced on a V100 (sm_70) with closely matching scores. Small differences are expected there because V100
  has no TF32. Other GPUs (A100, L40S, …) are untested.

## Licenses and citation

| Asset | License |
|---|---|
| OpenFold code | Apache-2.0 |
| OpenFold weights, OpenProteinSet | CC BY 4.0 |
| AlphaFold2 weights | CC BY 4.0 (outputs are for theoretical modeling, not clinical use) |
| PDB structures | CC0 1.0 |
| CAMEO target list | CAMEO terms of use; only target identifiers are used |

No gated or non-commercial assets are required. If you use this workflow, cite:

- Ahdritz, G. *et al.* OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. *Nature Methods* 21, 1514–1524 (2024). <https://doi.org/10.1038/s41592-024-02272-z>
- Ahdritz, G. *et al.* OpenProteinSet: Training data for structural biology at scale. NeurIPS 2023 Datasets & Benchmarks. arXiv:2308.05326
- Jumper, J. *et al.* Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021). <https://doi.org/10.1038/s41586-021-03819-2>
- Haas, J. *et al.* Continuous Automated Model EvaluatiOn (CAMEO) complementing the critical assessment of structure prediction in CASP12. *Proteins* 86, 387–398 (2018).

Related registry files: [model.yaml](model.yaml) ·
[recipes/openfold-inference-cameo.yaml](../../../../recipes/openfold-inference-cameo.yaml) ·
[benchmarks/results/openfold/openfold-ice-cameo.yaml](../../../../benchmarks/results/openfold/openfold-ice-cameo.yaml) ·
[resource-profiles/ice/ice-h100-h200.yaml](../../../../resource-profiles/ice/ice-h100-h200.yaml) ·
tracking issue [#11](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/11).

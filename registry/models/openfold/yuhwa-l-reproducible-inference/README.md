# OpenFold Reproducible Inference and Baseline Measurements

This recipe runs OpenFold structure inference reproducibly on **Georgia Tech PACE/ICE** for six proteins from 159 to 1,125 residues, using the official OpenFold code, the `model_1_ptm` parameters on the PACE shared filesystem, and precomputed alignments included here. One workflow covers all six:

1. **Setup** (section 1) and **data** (section 2): the environment, then the six proteins with their inputs, alignments, and experimental structures.
2. **Inference and validation** (sections 3–5): predict all six structures, check each against its experimental structure and the committed results, and render images.
3. **Baseline measurements** (section 6): inference time, GPU memory, GPU use, and accuracy for all six with OpenFold's default settings, as the starting point for optimization.

Settings:

| Item | Value |
|---|---|
| Proteins | Six, listed in `proteins.tsv` (section 2) |
| OpenFold commit | `be2ec1841f16c966c65ae0e7599ebbadc725757d` |
| Model / preset | `model_1_ptm` |
| Data random seed | 42 |
| Templates | Up to the day before each protein's experimental structure was released |
| Device | `cuda:0`, NVIDIA A100-PCIE-40GB (`coc-gpu`) |
| PyTorch | 2.5.1, CUDA 12.4 |
| Alignments | Precomputed (included) |
| Relaxation | Skipped |

Expected results (committed in `results/measurements/`):

| Protein | Residues | Mean pLDDT | TM-score | lDDT | RMSD (Å) |
|---|---|---|---|---|---|
| 9OUF_A | 159 | 87.47 | 0.946 | 0.924 | 2.98 |
| 6KWC_1 | 191 | 97.72 | 0.987 | 0.983 | 0.96 |
| 9Q5U_A | 325 | 91.71 | 0.877 | 0.868 | 2.67 |
| 9NXH_A | 500 | 91.48 | 0.720 | 0.940 | 12.85 |
| 9LB7_A | 777 | 95.73 | 0.990 | 0.967 | 1.12 |
| 8P5S_A | 1125 | 92.25 | 0.977 | 0.954 | 1.82 |

TM-score and lDDT range from 0 to 1 (1 is a perfect match with the experimental structure); RMSD is the average C-alpha distance after superposition. 9NXH's two domains are each predicted well, but placed differently relative to each other (section 6.3).

All commands below assume a PACE/ICE shell. First get this repository into scratch (login node); `$RECIPE` is this recipe's directory:

```bash
cd ~/scratch
git clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry
export RECIPE="$PWD/registry/models/openfold/yuhwa-l-reproducible-inference"
```

Until pull request #41 is merged, the recipe is only on the contributor's branch. Clone it with `git clone -b openfold-reproducible-inference https://github.com/Yuhwa-L/cybershuttle-ai4science-foundry.git` instead, then run the `cd` and `export` lines above.

In a new shell (for example, inside a later allocation), set `RECIPE` again before running the commands below.

## 1. Setup

Clone on the login node (1.1), then build the environment on a compute node (1.2–1.3). This is done once: the environment stays in scratch, and later allocations only activate it (section 3).

### 1.1 Clone OpenFold at the pinned commit (login node)

```bash
cd ~/scratch
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
git checkout be2ec1841f16c966c65ae0e7599ebbadc725757d
```

`~/scratch/openfold` is the default `OPENFOLD_DIR` used by `scripts/run_inference.sh`.

### 1.2 Create the Conda environment under scratch (compute node)

Get an interactive allocation first. Login nodes limit memory per user, and `mamba env create` there was killed with exit code 137 (out of memory) while solving the environment; the allocation's 32 GB is enough. Stay in this allocation through section 1.3, and set `RECIPE` again inside it.

```bash
salloc --account=coc --partition=coc-gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00
```

The environment is several GB. The PACE home quota is small, and creating it in `$HOME` failed for us with `OSError: [Errno 122] Disk quota exceeded`, so keep the Conda package cache, environments, and pip cache in scratch:

```bash
module load miniforge
export CONDA_PKGS_DIRS="$HOME/scratch/.conda/pkgs"
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

mamba env create -y -f "$RECIPE/environment.yml"
```

`flash-attn` is deliberately not in `environment.yml`: its build needs the already-installed PyTorch, and with it in the pip list `mamba env create` failed in our clean test. It is installed in the next step.

This creates `openfold-env` in `~/scratch/.conda/envs`. `CONDA_ENVS_PATH` must be set in every new shell before `conda activate openfold-env` (the SLURM script sets it for you).

### 1.3 Install OpenFold into the environment

```bash
conda activate openfold-env
pip install flash-attn==2.8.3 --no-build-isolation
unset GCC_ROOT
cd ~/scratch/openfold
scripts/install_third_party_dependencies.sh
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

`unset GCC_ROOT` is required on PACE: the default `gcc` module sets it, and with it set the Conda compiler fails with `cannot execute 'cc1plus'` while building OpenFold's CUDA attention kernel. The install script also prints a harmless `fatal: destination path 'cutlass' already exists` if it is run twice.

Check the installation:

```bash
python -c "import torch, flash_attn, openfold, cuda, tensorrt, polygraphy, pynvml; print(torch.__version__, torch.version.cuda)"
python ~/scratch/openfold/run_pretrained_openfold.py --help
```

`tensorrt`, `polygraphy`, and `cuda-python` are needed only because the OpenFold entry point imports them at this commit. TensorRT acceleration was not enabled. `pynvml` (from `nvidia-ml-py`) reads GPU memory and utilization in section 6.

Exact package versions from the original run are in `requirements-lock.txt` and `logs/environment.log`.

## 2. Proteins and data (nothing to download)

PACE/ICE provides the AlphaFold 2.3.2 data on its shared filesystem, and `scripts/run_inference.sh` uses it by default:

| Data | Path |
|---|---|
| Parameters | `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/params/params_model_1_ptm.npz` |
| Templates | `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb_mmcif/mmcif_files` |

Parameter SHA256: `5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84`

The six proteins are listed in `proteins.tsv`, which every script reads. Each has an input FASTA, precomputed alignments (`bfd_uniref_hits.a3m`, `hhsearch_output.hhr`, `mgnify_hits.sto`, `uniref90_hits.sto`), and its experimental structure:

| Protein | Residues | Released | Resolution | Description | Files |
|---|---|---|---|---|---|
| 9OUF_A | 159 | 2025-07-30 | 1.39 Å | ADP-ribosyl hydrolase | `length-series/` |
| 6KWC_1 | 191 | 2021-01-27 | 1.30 Å | Endo-1,4-beta-xylanase 2 | `inputs/`, `alignments/`, `reference/` |
| 9Q5U_A | 325 | 2025-09-24 | 2.50 Å | Carbohydrate-binding module of an RTX protein | `length-series/` |
| 9NXH_A | 500 | 2025-08-20 | 1.75 Å | Alpha-L-arabinofuranosidase | `length-series/` |
| 9LB7_A | 777 | 2025-07-30 | 2.35 Å | Trehalose-6-phosphate phosphorylase | `length-series/` |
| 8P5S_A | 1125 | 2023-08-16 | 2.46 Å | 2-oxoglutarate dehydrogenase | `length-series/` |

6KWC's alignments were taken from the 6KWC monomer example in the VizFold Foundation repository at commit `745229dcb0b7eb5ed06ebf6678e99eded418b543`. Its top templates are near-identical xylanases, so it is an easy target. The other five are newer proteins chosen to cover a range of lengths and to be new to the model; their alignments were computed on PACE (section 7). Because all alignments are included, **the MSA/database search does not need to be rerun**.

### 2.1 Unpack the alignments (login node)

The five newer proteins' alignments are stored compressed (35 MB in total), because they are large as plain text: 288 MB in total, and 138 MB for 8P5S alone, more than the 100 MB per-file limit on GitHub.

```bash
cd "$RECIPE/length-series/alignments"
for f in *.tar.gz; do tar -xzf "$f"; done
```

Each protein gets `length-series/alignments/<ID>_A/` with the same four files as `alignments/6KWC_1/`.

## 3. Run inference on all six proteins (GPU node, about 20 minutes)

The 20 minutes is estimated from section 6.3: about 9 minutes of forward passes in total (6 minutes for 8P5S alone), plus model loading and feature preparation for each protein.

Inference needs a GPU, so do not run it on a login node.

### Option A: batch job

```bash
cd "$RECIPE"
sbatch slurm/openfold_inference.slurm
```

Submit from the recipe directory. The script requests one A100 on `coc-gpu` (account `coc`, 8 CPUs, 64 GB, 2 hours), activates `openfold-env` from `~/scratch/.conda/envs`, prints `nvidia-smi`, and calls `scripts/run_inference.sh`. Edit the `#SBATCH` lines if your account or partition differs. The log is `openfold-inference-<job ID>.out`.

### Option B: interactive allocation

```bash
salloc --account=coc --partition=coc-gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00

module load miniforge
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
conda activate openfold-env
cd "$RECIPE"
./scripts/run_inference.sh
```

`run_inference.sh` checks that the OpenFold checkout, inputs, unpacked alignments, parameters, templates, and a CUDA GPU are available before starting. It then predicts each protein in `proteins.tsv`, shortest first, with `model_1_ptm` on `cuda:0`, `--data_random_seed 42`, precomputed alignments, and `--skip_relaxation`. Each protein's `--max_template_date` is the day before its experimental structure was released, so OpenFold cannot use that structure as a template. To predict only some proteins, name them: `./scripts/run_inference.sh 6KWC_1 9OUF_A`.

Output goes to `results/run/<protein>/predictions/<protein>_model_1_ptm_unrelaxed.pdb`. Set `OUTPUT_DIR` to write somewhere else; `OPENFOLD_DIR`, `PARAMS`, and `TEMPLATE_DIR` can be overridden the same way.

## 4. Validate

```bash
cd "$RECIPE"
python scripts/validate_structure.py
```

For each protein in `results/run/`, the script checks the predicted sequence against the FASTA, reads pLDDT (stored in the B-factor column), compares the prediction with experimental chain A (TM-score, C-alpha lDDT, C-alpha RMSD; `scripts/structure_metrics.py`), and writes `<protein>_superposed_on_reference.pdb` (the prediction rotated onto the experimental structure). It then compares the numbers with the committed results, within 0.5 pLDDT, 0.01 TM-score and lDDT, and 0.1 Å RMSD, and writes them to `results/run/validation.csv`. Expected output:

```text
Protein  Residues   Seq  pLDDT     TM   lDDT   RMSD  Result
9OUF_A        159  True  87.47  0.946  0.924   2.98  ok  (baseline 87.47 / 0.946 / 0.924 / 2.98)
6KWC_1        191  True  97.72  0.987  0.983   0.96  ok  (baseline 97.72 / 0.987 / 0.983 / 0.96)
9Q5U_A        325  True  91.71  0.877  0.868   2.67  ok  (baseline 91.71 / 0.877 / 0.868 / 2.67)
9NXH_A        500  True  91.48  0.720  0.940  12.85  ok  (baseline 91.48 / 0.720 / 0.940 / 12.85)
9LB7_A        777  True  95.73  0.990  0.967   1.12  ok  (baseline 95.73 / 0.990 / 0.967 / 1.12)
8P5S_A       1125  True  92.25  0.977  0.954   1.82  ok  (baseline 92.25 / 0.977 / 0.954 / 1.82)
ALL PASS
```

To check the committed predictions themselves, run `python scripts/validate_structure.py --results results/measurements`. The committed 6KWC prediction is also at `results/6KWC_1_model_1_ptm_unrelaxed.pdb` (SHA256 `eca9dedb426047a781658bf8679e5f99d26fcb007e8f4c5a7929bfd15cd48b69`, the same file as in `results/measurements/6KWC_1/predictions/`).

## 5. Visualizations (optional)

PyMOL is not in `openfold-env`. Create a small separate environment once, on a compute node (the login node may run out of memory):

```bash
module load miniforge
export CONDA_PKGS_DIRS="$HOME/scratch/.conda/pkgs"
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
mamba create -y -n pymol-env -c conda-forge pymol-open-source=3.1.0 python=3.11
```

Then render all six predictions (after section 4):

```bash
cd "$RECIPE"
~/scratch/.conda/envs/pymol-env/bin/pymol -cq scripts/render_structures.py -- results/run visualizations/run
```

For each protein this writes `<protein>_plddt.png` (prediction colored by pLDDT, red below 50, green at 100) and `<protein>_openfold_vs_experimental.png` (prediction in blue on the experimental chain in gray). The committed images are in `visualizations/measurements/`. PyMOL's ray tracing is not bit-for-bit repeatable, so new images can differ slightly in pixels from the committed ones.

## 6. Measure inference time, GPU use, and accuracy

This is the baseline for later optimization: the same OpenFold settings as section 3 (default attention and chunking, nothing tuned), run on the six proteins. It needs the environment from section 1 and the unpacked alignments from step 2.1, and gives the same predictions as section 3 plus time and GPU measurements.

### 6.1 Run the measurements (GPU node, about 45 minutes)

```bash
cd "$RECIPE"
sbatch slurm/measure_inference.slurm
```

The job runs `scripts/measure_inference.py` once per protein, from shortest to longest, on one A100 40GB (`coc-gpu`, 8 CPUs, 64 GB, 3-hour limit). `coc-gpu` has both 40GB and 80GB A100s, and time and memory differ between them (section 6.3), so the script asks for the 40GB type with `--constraint=A100-40GB`, the card the baseline was measured on. Only two nodes have it, so the job may wait longer; remove that line to accept either A100, but then compare times only with runs on the same card. For each protein, OpenFold loads the model once, runs one unmeasured warm-up pass, then three measured passes. Each protein's `--max_template_date` is the day before its experimental structure was released, so OpenFold cannot use that structure as a template.

What is measured, per protein:

- **Time**: one model forward pass on the GPU (median of the three passes). Model loading and input feature preparation are not included.
- **GPU memory**: peak memory PyTorch allocated, and peak memory used on the whole card (read from the driver every 0.1 s, including PyTorch's cache).
- **GPU use**: the driver's utilization percentage, averaged over the measured passes.
- **Confidence** from OpenFold: mean pLDDT, pTM, and mean PAE.
- **Accuracy** against the experimental chain A, over the residues it resolves: TM-score, C-alpha lDDT, and C-alpha RMSD (`scripts/structure_metrics.py`).

Output goes to `results/measurements-<job ID>/`: a `measurements.csv` table with one row per protein (also printed at the end of the job log, `openfold-measure-<job ID>.out`), and a folder per protein with the prediction, the prediction superposed on the experimental structure, per-residue pLDDT and the PAE matrix (`confidence.npz`), and `result.json` with every pass and the exact OpenFold command. To measure one protein by itself, run `python scripts/measure_inference.py --help` inside a GPU allocation.

### 6.2 Render images (optional, any node)

With `pymol-env` from section 5, render every protein of a measurement run (replace `<job ID>`):

```bash
cd "$RECIPE"
~/scratch/.conda/envs/pymol-env/bin/pymol -cq scripts/render_structures.py -- results/measurements-<job ID> visualizations/measurements-<job ID>
```

The images are the same two per protein as in section 5.

### 6.3 Baseline results

Committed in `results/measurements/` and `visualizations/measurements/`, from job 6886502 (2026-11-09, one NVIDIA A100-PCIE-40GB, log `logs/measurements-6886502.out`):

| Protein | Residues | Time (s) | Peak allocated (GiB) | Peak card memory (GiB) | GPU use | Mean pLDDT | pTM | TM-score | lDDT | RMSD (Å) |
|---|---|---|---|---|---|---|---|---|---|---|
| 9OUF_A | 159 | 6.1 | 3.7 | 6.0 | 93% | 87.5 | 0.77 | 0.95 | 0.92 | 2.98 |
| 6KWC_1 | 191 | 7.8 | 4.4 | 7.5 | 95% | 97.7 | 0.93 | 0.99 | 0.98 | 0.96 |
| 9Q5U_A | 325 | 18.8 | 7.2 | 17.5 | 97% | 91.7 | 0.88 | 0.88 | 0.87 | 2.67 |
| 9NXH_A | 500 | 39.8 | 11.0 | 37.3 | 97% | 91.5 | 0.69 | 0.72 | 0.94 | 12.85 |
| 9LB7_A | 777 | 110.2 | 23.3 | 39.3 | 98% | 95.7 | 0.94 | 0.99 | 0.97 | 1.12 |
| 8P5S_A | 1125 | 350.5 | 10.7 | 36.7 | 97% | 92.3 | 0.89 | 0.98 | 0.95 | 1.82 |

The three measured passes of each protein were within 3% of each other (within 1% for all but 8P5S). The whole job took 44 minutes.

- **Time grows much faster than length**: 8P5S is about 6 times longer than 6KWC but takes about 45 times longer.
- **Memory**: from 500 residues on, the card is nearly full (37–39 of 40 GiB), mostly memory PyTorch keeps cached. During the warm-up pass OpenFold picks a chunk size that fits in memory, which is why 8P5S allocates less than 9LB7. Proteins much longer than 1,125 residues may not fit on a 40 GB A100.
- **GPU use** stays above 93%, so the GPU is busy for the whole forward pass.
- **Accuracy** is high for all but 9NXH. Each of its two domains is predicted well (lDDT 0.94), but they are placed differently relative to each other, and a disordered tail sticks out, so the whole-chain TM-score and RMSD are worse. OpenFold's own pTM (0.69, the lowest here) shows the same uncertainty.

A rerun on the same GPU model should give times within a few percent and the same accuracy numbers. On an A100 80GB (job 6893957, section "What was tested") every accuracy and confidence number above was the same, times were 7–10% shorter, and the longer proteins used more memory (66 GiB for 8P5S), because OpenFold picks larger chunks when more memory is free. That also changes 8P5S's coordinates very slightly (0.009 Å on average, 0.25 Å at most); the other five predictions were byte-identical.

## 7. How the proteins were prepared (optional)

The inputs, experimental structures, and alignments of the five newer proteins are already in the repository. This section records how they were chosen and made, and how to regenerate them. Their selection details are also in `length-series/targets.tsv`.

### 7.1 Selection rules

They were chosen from the PDB copy on the PACE shared filesystem (snapshot of 2025-09-30). For each target length of 150, 300, 500, 800, and 1,100 residues, the newest entries within 8% of that length were checked against these rules, a few dozen per length. Near 300 residues all of those failed the last rule, so the range was widened to 17% there.

- one protein chain per entry, no unknown residues;
- X-ray structure at 2.5 Å or better, with at least 90% of residues resolved;
- first released in 2023 or later;
- new to the model: none of about 40 evenly spaced 20-residue pieces of the sequence appears in any older PDB entry (IDs starting with 1–6, deposited up to about 2020). The AlphaFold parameters used here were trained on structures released before May 2018, so a protein that passes shares no long identical stretch with the training data (more distant relatives can still exist; the alignments in step 7.3 find those). Candidates that failed this rule, such as 9UOA (330 older entries share a piece), were dropped.

One protein per length was then picked by hand from the candidates that passed every rule. To confirm that the committed proteins meet every rule (login node, about 30 seconds; the script needs only Python 3):

```bash
cd "$RECIPE"
python scripts/check_length_series.py
```

Each protein should print `PASS`, with 0 older entries sharing a piece.

### 7.2 Regenerate inputs (optional, login node)

```bash
cd "$RECIPE"
python scripts/prepare_length_series.py
```

This writes `length-series/inputs/<ID>_A.fasta` (the chain's full sequence) and copies the experimental structure to `length-series/reference/<ID>.cif`, both from the PACE PDB copy. Nothing is downloaded.

### 7.3 Recompute alignments (optional, CPU nodes, about 6 hours)

```bash
cd "$RECIPE"
sbatch slurm/compute_alignments.slurm
```

This runs OpenFold's `scripts/precompute_alignments.py` once per protein, as a 5-task job array on `coc-cpu` (8 CPUs and 64 GB each), against the AlphaFold databases on PACE: UniRef90 and MGnify (JackHMMER), BFD and UniRef30 (HHblits), and PDB70 (HHsearch, template hits). It writes the same `length-series/alignments/<ID>_A/` folders as step 2.1. A protein whose folder already exists is skipped, so delete a folder to recompute it. To pack the results again:

```bash
cd "$RECIPE/length-series/alignments"
for d in */; do d=${d%/}; tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-07 00:00Z' -cf - "$d" | gzip -n > "$d.tar.gz"; done
```

The committed alignments come from job 6852413 (2026-11-06, logs in `logs/length-series-alignments/`), which ran all five tasks in parallel and finished `COMPLETED` in under 6 hours:

| Protein | Wall time | Peak memory | Alignments | Compressed |
|---|---|---|---|---|
| 8P5S_A | 5 h 49 min | 38 GB | 138 MB | 15 MB |
| 9LB7_A | 1 h 49 min | 41 GB | 91 MB | 13 MB |
| 9NXH_A | 2 h 43 min | 41 GB | 56 MB | 6.3 MB |
| 9OUF_A | 2 h 42 min | 37 GB | 0.8 MB | 0.2 MB |
| 9Q5U_A | 2 h 40 min | 38 GB | 2.0 MB | 0.3 MB |

Wall time depends on the shared file system as much as on length. Peak memory stays near 40 GB for every protein, so keep `--mem=64G`. 9OUF and 9Q5U find few related sequences (small folders), which makes them useful low-MSA test cases. A rerun can find slightly different hits if the PACE databases are updated.

## What was tested

Runs 1–8 below used the recipe's earlier layout, in which 6KWC was predicted and validated first (old sections 3–5) and the other five proteins were added afterwards (old sections 6–7). Section numbers in runs 1–8 refer to that layout. The current section 3 runs the same OpenFold command per protein, with the same settings and template dates as the measurement job (section 6), which gave a 6KWC prediction byte-identical to the original 6KWC run (run 6). `scripts/validate_structure.py` was checked on the committed predictions (`--results results/measurements`, all six PASS, superposed structures byte-identical); the six-protein `run_inference.sh` itself has not yet been run end to end on a GPU.

Everything below was run on PACE/ICE (`coc-gpu`). Runs 1–4 were on 2026-10-11 with one NVIDIA A100-PCIE-40GB; the later runs give their own date and GPU.

1. **Original run** (existing environment): inference exit status 0 (`logs/inference.log`), validation in `logs/validation.log`.
2. **Clean-environment rerun** (`openfold-repro-test`, built under `~/scratch/.conda/envs`): inference was run through `slurm/openfold_inference.slurm` with a separate `OUTPUT_DIR`. OpenFold exited with status 0 (`logs/clean-reproduction.log`, `logs/clean-reproduction-validation.log`).
3. **Guide walkthrough**: sections 1.2, 1.3, 3 (Option B commands), and 4 were then run exactly as written, in a shell without any personal Conda initialization, creating `openfold-env` with a single `mamba env create`. The import check printed `2.5.1 12.4`, `run_pretrained_openfold.py --help` worked, and inference exited with status 0 (`logs/guide-walkthrough.log`).

4. **Batch job**: `sbatch slurm/openfold_inference.slurm` was submitted from the recipe directory as in section 3, Option A. Job 6200577 finished `COMPLETED` with exit code 0:0 in 4 min 18 s wall clock, using about 2.8 GB of memory (`logs/sbatch-6200577.out`, `logs/sbatch-6200577.err`).

Runs 1–4 all produced a **byte-identical** prediction (SHA256 above) and the same validation numbers.

5. **Fresh check** (2026-11-08): `slurm/openfold_inference.slurm` and section 4 were run again on job 6884808, which was given an NVIDIA A100 80GB PCIe. The prediction was again byte-identical (`logs/final-check-6884808.out`).
6. **Measurements** (2026-11-09): section 7 on job 6886502, with all six proteins finishing with status `ok` (section 7.3). The 6KWC prediction there has the same RMSD and pLDDT as above.
7. **Clean rerun of the whole guide** (2026-11-09, job 6893957, one A100 80GB, 56 minutes): a fresh clone of this repository and of OpenFold, a new environment built with an empty pip cache, then sections 1.2, 1.3, 3 (Option A), 4, 6.1, 7.1, 7.2, and 5 (`logs/clean-rerun-6893957.out`). Every step exited with status 0. The 6KWC prediction was byte-identical, the unpacked alignments matched the originally computed ones, and the measurements matched section 7.3 as described there. The new environment resolved some newer packages than `requirements-lock.txt` (for example `pytorch-lightning` 2.6.5 instead of 1.9.4) without changing the results.
8. **Second clean rerun, from GitHub** (2026-11-15, job 7077745, one A100 40GB, 65 minutes): this repository cloned from GitHub as in "Before you start", a fresh OpenFold clone, a new environment with an empty pip cache, then sections 6.1, 6.2, and `scripts/check_length_series.py` on a login node, and sections 1.2, 1.3, 3 (Option A), 4, 7.1, 7.2, and 5 in one batch job (`logs/clean-rerun-7077745.out`). Section 6.3's alignment programs (`jackhmmer`, `hhblits`, `hhsearch`, `kalign`) and `precompute_alignments.py --help` were checked to start, without the six-hour alignment run, and the checker printed PASS for all five proteins. Every step exited with status 0. Section 6.2 rewrote the committed inputs and references byte for byte, all six predictions and superposed structures were byte-identical to `results/measurements/` (8P5S included, since this run used the same 40GB card), the accuracy numbers were identical, and every median time was within 1% of section 7.3.

Notes:

- The environment resolved to the latest compatible package versions on that date, which may differ from `requirements-lock.txt`.
- The 6KWC prediction was byte-identical on both A100 types (40GB and 80GB). Other GPUs or package versions may give small numerical differences; use the validation numbers, not the checksum, to judge a reproduction.

## Troubleshooting

These are the failures we hit on PACE/ICE while building this recipe, and how each was resolved.

| Symptom | Cause | Fix |
|---|---|---|
| `mamba env create` killed with exit code 137 | The login node's per-user memory limit is too low for the environment solver | Run section 1.2 inside a `salloc` allocation with `--mem=32G` |
| `OSError: [Errno 122] Disk quota exceeded` during `conda env create` | Conda package cache and environment were written to `$HOME` | Set `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH`, `PIP_CACHE_DIR`, and `TMPDIR` to scratch (section 1.2) |
| `Pip failed` at `flash-attn` ("Getting requirements to build wheel ... error") during `mamba env create` | `flash-attn` needs PyTorch at build time, which pip's isolated build does not see | Install it afterwards with `pip install flash-attn==2.8.3 --no-build-isolation` (section 1.3) |
| `fatal error: cannot execute 'cc1plus'` and `nvcc fatal: Failed to preprocess host compiler properties` in `install_third_party_dependencies.sh` | The default PACE `gcc` module sets `GCC_ROOT`, which misdirects the Conda compiler | `unset GCC_ROOT` before running the script (section 1.3) |
| `ModuleNotFoundError: No module named 'attn_core_inplace_cuda'` | The previous step failed, so OpenFold's CUDA kernel was not built | Fix the compiler error above and rerun the install script |
| `EnvironmentNameNotFound: openfold-env` | `CONDA_ENVS_PATH` is not set in the current shell | `export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"` |
| `jackhmmer: error while loading shared libraries: libopenblas.so.0` in the alignment job | An environment created before `libopenblas` was added to `environment.yml` | On a compute node: `mamba install -n openfold-env --freeze-installed -c conda-forge libopenblas` |
| `run_inference.sh` stops with `ERROR: No CUDA GPU visible` | Running on a login node | Run inside a GPU allocation (section 3) |
| `run_inference.sh` stops with `ERROR: Missing alignments: .../length-series/alignments/<ID>_A` | The compressed alignments were not unpacked | Run step 2.1 |

Warnings about `libaio`, `torch.cuda.amp` deprecation, and the Triton cache directory appear in successful runs and can be ignored.

## Runtime and resources

On one A100-PCIE-40GB with 8 CPUs and 32 GB of memory, model inference for 6KWC (191 residues) took about 8 to 17 seconds, and the whole OpenFold command of the original run took 1 min 41 s of wall-clock time. That run peaked at about 2.1 GB of host memory (`logs/inference.log`). Creating the environment took about 6 minutes with a warm package cache, and about 15 minutes with an empty pip cache (job 6893957). For all six proteins, inference times are in section 6.3; alignment times for the five newer proteins are in section 7.3.

## Cleanup

```bash
rm -rf "$RECIPE/results/run" "$RECIPE/visualizations/run"   # your own predictions and images
rm -rf "$RECIPE"/results/measurements-* "$RECIPE"/visualizations/measurements-*   # your measurement runs
rm -rf "$RECIPE"/length-series/alignments/*/          # unpacked alignments (the .tar.gz files stay)
conda env remove -n openfold-env                      # the environment (needs CONDA_ENVS_PATH set)
conda env remove -n pymol-env                         # the PyMOL environment, if created
rm -rf ~/scratch/.conda/pkgs ~/scratch/.cache/pip     # package caches
rm -rf ~/scratch/openfold                             # the OpenFold checkout
```

## Files

| Path | Contents |
|---|---|
| `model.yaml` | Registry entry for OpenFold (moved here from `registry/models/biology/openfold/`) |
| `environment.yml` | Conda environment specification |
| `requirements-lock.txt` | Package versions of the original run |
| `proteins.tsv` | The six proteins: input, alignment, and reference paths, release dates |
| `scripts/run_inference.sh` | Predicts all six proteins, with preflight checks |
| `scripts/validate_structure.py` | Sequence, pLDDT, TM-score, lDDT, and RMSD check of all six against the committed results |
| `slurm/openfold_inference.slurm` | PACE/ICE batch script |
| `scripts/prepare_length_series.py` | Writes length-series inputs and reference structures |
| `scripts/check_length_series.py` | Confirms each length-series protein meets the selection rules |
| `slurm/compute_alignments.slurm` | Computes length-series alignments on CPU nodes |
| `inputs/`, `alignments/`, `reference/` | 6KWC input, alignments, experimental structure |
| `length-series/` | The five newer proteins: selection list, inputs, compressed alignments, experimental structures |
| `logs/length-series-alignments/` | Logs of the alignment job (6852413) |
| `logs/clean-rerun-6893957.out` | Log of the clean rerun of the whole guide |
| `logs/clean-rerun-7077745.out` | Log of the second clean rerun, from a GitHub clone on an A100 40GB |
| `slurm/measure_inference.slurm` | Measures all six proteins on one GPU (section 6) |
| `scripts/measure_inference.py` | Times inference and records GPU memory, GPU use, and accuracy for one protein |
| `scripts/structure_metrics.py` | TM-score, lDDT, RMSD, and superposition |
| `scripts/render_structures.py` | PyMOL images of an inference or measurement run |
| `scripts/visualize_structure.pml` | PyMOL script for the original 6KWC images in `visualizations/` |
| `results/measurements/`, `visualizations/measurements/` | Baseline measurements and images (job 6886502) |
| `logs/` | Environment, GPU, inference, and validation logs |
| `results/`, `reference/`, `visualizations/` | Reference prediction, experimental structure, images |

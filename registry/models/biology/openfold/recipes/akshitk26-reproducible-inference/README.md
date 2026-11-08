# OpenFold monomer inference (6KWC_1) on PACE ICE

This recipe runs pretrained OpenFold v2.2.0 inference for the protein `6KWC_1` on one GPU of Georgia Tech PACE ICE. Then it checks the predicted structure against the experimental structure from the RCSB PDB.

You do not need a `.env` file, the AWS CLI, an OpenFold module, or any username-specific path. The scripts handle the environment, the build, and the downloads. Large files go to your scratch directory (`~/scratch`), not to your home directory.

## Expected result

| item | value from the reference run (job `6190401`) |
|---|---|
| source | OpenFold `v2.2.0`, commit `e938c184a291bf053af3b14c1e3e8bb29aee57e2` |
| model | `model_3_ptm` with `finetuning_no_templ_ptm_1.pt` |
| input | `inputs/6KWC_1.fasta` with OpenFold's template-free example alignments |
| hardware | 1 x Quadro RTX 6000, 4 CPUs, 32 GB RAM |
| inference time | 26.21 seconds |
| peak GPU memory | 6433 MiB |
| output | PDB with 191 residues and 1482 atoms |
| mean confidence (pLDDT) | 96.66 |
| match to RCSB `6KWC` chain A | 100% sequence identity, 1.154 Å CA RMSD |

The reference output is committed at `results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb`. A new run writes to `results/run-<job id>/` and leaves the reference output alone.

## What you need

- A PACE ICE account. Off campus, connect to the GT VPN first.
- Time for the one-time install. Most of it goes to solving the conda env and building the CUDA extension. After that, a run takes a few minutes (the reference job took 78 seconds).
- Disk space in `~/scratch` for the conda env, the OpenFold source build, and the 373 MB weights file. Check your quota with `pace-quota`.

## Step 1: log in and get the code

```bash
ssh <your-gt-username>@login-ice.pace.gatech.edu

cd ~/scratch
git clone --branch add-openfold-inference https://github.com/akshitk26/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry/registry/models/biology/openfold/recipes/akshitk26-reproducible-inference
```

After this PR is merged, clone `https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git` instead, without `--branch`.

Run every later command from this recipe directory.

## Step 2: get a GPU node for the one-time install

```bash
salloc --account=coc --partition=ice-gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gpus=rtx_6000:1 --time=02:00:00
```

`coc` is the charge account this recipe was tested with. If `salloc` rejects it, list your accounts with `sacctmgr -nP show assoc where user=$USER format=account` and use one of those.

`--gpus=rtx_6000:1` asks for an NVIDIA RTX 6000. Always name an NVIDIA type: the `ice-gpu` partition also has AMD GPUs (`mi210`, `r9700`), and OpenFold cannot run on them. Other NVIDIA types on ICE (`v100`, `a40`, `a100`, `l40s`, `h100`, `h200`) also work. List them with `sinfo -p ice-gpu -h -o "%G" | sort -u`.

When the prompt changes to a compute node, `cd` back to the recipe directory if you are not already there.

## Step 3: install the environment and build OpenFold

```bash
bash scripts/install_environment.sh
```

This script:

1. loads the ICE modules `gcc/12.3.0`, `cuda/12.1.1` and `anaconda3`,
2. creates the conda env from `environment.yml` at `~/scratch/openfold-6kwc/conda-env`,
3. clones OpenFold at the pinned commit and CUTLASS at the pinned commit,
4. builds OpenFold's CUDA extension with the ICE module compiler.

It ends with `install finished`. If it stops partway, run it again. Steps that already finished are skipped.

## Step 4: download the weights and inputs

```bash
bash scripts/download_artifacts.sh
```

This downloads the weights over HTTPS from OpenFold's public bucket and checks their SHA-256 against the reference run. It copies the precomputed alignments from the OpenFold source and downloads the four mmCIF files OpenFold needs at startup. It ends with `downloads finished`.

## Step 5: check the setup

```bash
bash scripts/setup_check.sh
```

You should see the GPU name and then `setup check passed`. This step needs a GPU, so run it on the node from step 2.

## Step 6: run inference as a batch job

Leave the interactive node (`exit`), go back to the recipe directory, and submit:

```bash
sbatch jobs/openfold_inference.sbatch
# if you used a different account in step 2:
# sbatch --account=<your-account> jobs/openfold_inference.sbatch
# if RTX 6000 GPUs are busy, any NVIDIA type works:
# sbatch --gpus=v100:1 jobs/openfold_inference.sbatch
```

Check on it with `squeue -u $USER`. When it finishes, read the log:

```bash
cat evidence/slurm-<job id>.out
```

The job runs the setup check, inference, and both validation scripts. The end of the log should show:

```text
unique residues: 191
mean pdb b-factor / predicted confidence: 96.66
sequence identity across alignment: 100.00%
ca rmsd after superposition (angstrom): 1.154
```

GPU type and driver can change the last digits of confidence and RMSD a little. A run is good if it has 191 residues, 100% identity, confidence above 90, and CA RMSD below 2 Å.

## Outputs

| path | contents |
|---|---|
| `results/run-<job id>/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb` | predicted structure, pLDDT in the B-factor column |
| `evidence/slurm-<job id>.out` | full job log |
| `evidence/validation-<job id>.log` | file check, residue count, mean confidence |
| `evidence/structure-comparison-<job id>.log` | sequence identity and CA RMSD against RCSB `6KWC` |
| `evidence/gpu-memory-<job id>.csv` | GPU memory sampled every second |
| `evidence/platform-<job id>.txt` | node, modules, `nvidia-smi` |

## Changing paths

Every path has a default in `scripts/common.sh`. To use other locations, copy `.env.example` to `.env` in this directory and edit it. The most useful setting is `OPENFOLD_WORK`, which moves all large files at once.

## Cleanup

```bash
rm -rf ~/scratch/openfold-6kwc
```

This deletes the env, the OpenFold build, and the downloads. Your results in the recipe directory stay.

## Records

| path | contents |
|---|---|
| `docs/CLUSTER_COMMANDS.md` | what each script runs, and troubleshooting |
| `environment.yml` and `evidence/conda-explicit.txt` | environment spec and the exact conda package set from the reference run |
| `inputs/MANIFEST.md` and `docs/PROVENANCE.md` | sources, versions, and checksums |
| `docs/EXECUTION_RECORD.md` | batch resources, runtime, output, and earlier failed jobs |
| `docs/VALIDATION.md` | smoke test and structure validation |
| `evidence/` | logs, hashes, and resource records from the reference run |

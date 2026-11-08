# what the scripts run

The README is the step-by-step guide. This page lists what each script does, so you can run a step by hand or debug it. No path here depends on a username: `$OPENFOLD_WORK` defaults to `$HOME/scratch/openfold-6kwc`, and the recipe directory is wherever you cloned the repository.

## scripts/common.sh

Every other script sources this file. It:

- reads `.env` from the recipe directory if it exists (optional),
- sets defaults for all paths (`OPENFOLD_WORK`, `OPENFOLD_ENV`, `OPENFOLD_DIR`, `ALIGNMENT_DIR`, `TEMPLATE_MMCIF_DIR`, `CHECKPOINT_PATH`, `FASTA_DIR`, `OUTPUT_DIR`, `CONFIG_PRESET`),
- loads the ICE modules:

```bash
module purge
module load gcc/12.3.0
module load cuda/12.1.1
module load anaconda3
```

- activates the env with `conda activate "$OPENFOLD_ENV"`, turning off `set -u` around activation, because the conda gcc activation script reads unset variables (`SYS_SYSROOT`, see `evidence/slurm-6189984.err`),
- sets `TMPDIR` and `TRITON_CACHE_DIR` under `$OPENFOLD_WORK` so temporary files stay off the home file system.

## scripts/install_environment.sh

```bash
# compiler from the gcc/12.3.0 module, recorded before conda adds its own gcc to PATH
MODULE_CC="$(command -v gcc)"; MODULE_CXX="$(command -v g++)"

CONDA_PKGS_DIRS="$OPENFOLD_WORK/conda-pkgs" conda env create -f environment.yml -p "$OPENFOLD_ENV"
conda activate "$OPENFOLD_ENV"
python -m pip install --no-build-isolation flash-attn==2.8.3.post1

git clone https://github.com/aqlaboratory/openfold.git "$OPENFOLD_DIR"
git -C "$OPENFOLD_DIR" checkout e938c184a291bf053af3b14c1e3e8bb29aee57e2   # v2.2.0

cd "$OPENFOLD_DIR"
wget -N --no-check-certificate -P openfold/resources \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1 cutlass   # bf9da7b76c766d7ee7d536afc77880a4ef1f1156
conda env config vars set CUTLASS_PATH="$OPENFOLD_DIR/cutlass" KMP_AFFINITY=none

export CC="$MODULE_CC" CXX="$MODULE_CXX" CUDAHOSTCXX="$MODULE_CXX"
export CUTLASS_PATH="$OPENFOLD_DIR/cutlass"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"   # all NVIDIA GPU types on ICE
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
MAX_JOBS=4 python setup.py install
```

The build uses the compiler from the `gcc/12.3.0` module, found with `command -v gcc` after `module load`. Earlier versions of this page used a fixed Spack install path, which only worked on one cluster image.

## scripts/download_artifacts.sh

```bash
# weights over HTTPS (no AWS CLI needed), then a sha-256 check
curl -fL --retry 3 -o "$CHECKPOINT_PATH" \
  https://openfold.s3.amazonaws.com/openfold_params/finetuning_no_templ_ptm_1.pt
# expected: 1a179937a6f61143c41f8b6a0b763ac1f4762c9dc63d2fb151240930c110bee1

# template-free alignments: the two MSA files, without hhsearch_output.hhr
mkdir -p "$ALIGNMENT_DIR/6KWC_1"
cp "$OPENFOLD_DIR"/examples/monomer/alignments/6KWC_1/{mgnify_hits.sto,uniref90_hits.sto} "$ALIGNMENT_DIR/6KWC_1/"

# mmCIF files the OpenFold runner needs at startup
for pdb in 2dfb 3akq 1te1 1h1a; do
  curl -fL --retry 3 -o "$TEMPLATE_MMCIF_DIR/$pdb.cif" "https://files.rcsb.org/download/$pdb.cif"
done
```

`model_3_ptm` disables templates. The alignment copy has no `hhsearch_output.hhr`, so the v2.2 data pipeline creates empty template features. The mmCIF directory is still required because the OpenFold runner checks it when it starts.

If you have the AWS CLI (it is in `environment.yml`), this command fetches the same file:

```bash
aws s3 cp --no-sign-request --region us-east-1 \
  s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt "$CHECKPOINT_PATH"
```

## scripts/run_inference.sh

```bash
cd "$OPENFOLD_DIR"
python run_pretrained_openfold.py "$FASTA_DIR" "$TEMPLATE_MMCIF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_precomputed_alignments "$ALIGNMENT_DIR" \
  --config_preset model_3_ptm \
  --openfold_checkpoint_path "$CHECKPOINT_PATH" \
  --model_device "cuda:0" \
  --data_random_seed 0 \
  --skip_relaxation
```

## validation (run by the batch job, or by hand)

These two scripts need only Python and NumPy, so they also run without a GPU:

```bash
python scripts/validate_output.py results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb
python scripts/compare_ca_rmsd.py \
  results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb A \
  inputs/reference/6kwc.pdb A
```

`inputs/reference/6kwc.pdb` is committed and came from https://files.rcsb.org/download/6KWC.pdb.

## recording artifact hashes

```bash
git -C "$OPENFOLD_DIR" rev-parse HEAD > evidence/openfold-commit.txt
conda list --explicit > evidence/conda-explicit.txt
sha256sum inputs/6KWC_1.fasta > evidence/input-sha256.txt
sha256sum "$CHECKPOINT_PATH" > evidence/checkpoint-sha256.txt
sha256sum "$TEMPLATE_MMCIF_DIR"/*.cif > evidence/template-mmcif-sha256.txt
```

## troubleshooting

| symptom | cause and fix |
|---|---|
| `the 'module' command is not available` | you are not on a PACE ICE node; log in to `login-ice.pace.gatech.edu` |
| `conda env not found` | run `bash scripts/install_environment.sh` first |
| `missing path: ...` | run `bash scripts/download_artifacts.sh` |
| job fails in seconds with `nvidia-smi: command not found` (exit 127) | the job got an AMD GPU (`mi210` or `r9700`); request an NVIDIA type, e.g. `--gpus=rtx_6000:1` |
| `cuda unavailable` | you are on a login node; use `salloc` (README step 2) or `sbatch` |
| `salloc`/`sbatch` rejects `--account=coc` | find your account with `sacctmgr -nP show assoc where user=$USER format=account` and pass `--account=<it>` |
| `~/scratch not found` | set `OPENFOLD_WORK` in `.env` to a directory on scratch storage |
| `checkpoint checksum mismatch` | partial download; delete the `.part` file and rerun the download script |

## cleanup

```bash
scancel <job id>          # only if a job is still running
rm -rf "$HOME/scratch/openfold-6kwc"
```

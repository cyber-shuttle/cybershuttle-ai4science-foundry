# ice workflow

## gpu allocation

```bash
salloc --account=coc --partition=ice-gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gpus=1 --time=02:00:00
```

## modules and conda

```bash
module purge
module load gcc/12.3.0
module load cuda/12.1.1
module load anaconda3
```

## project and source

```bash
PROJECT_DIR="$HOME/openfold-reproducible-inference"
git clone https://github.com/akshitk26/openfold-reproducible-inference.git "$PROJECT_DIR"
cd "$PROJECT_DIR"
mkdir -p artifacts/params artifacts/template_mmcif artifacts/alignments_no_templates/6KWC_1 evidence results/run-001

git clone https://github.com/aqlaboratory/openfold.git artifacts/openfold
git -C artifacts/openfold checkout v2.2.0
git -C artifacts/openfold rev-parse HEAD
```

## environment

```bash
conda env create -f environment.yml
conda activate openfold-repro

mkdir -p "$HOME/openfold-tmp"
TMPDIR="$HOME/openfold-tmp" \
  python -m pip install --no-build-isolation flash-attn==2.8.3.post1
```

## openfold extension

```bash
cd artifacts/openfold

wget -N --no-check-certificate -P openfold/resources \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
mkdir -p tests/test_data/alphafold/common
ln -rs openfold/resources/stereo_chemical_props.txt tests/test_data/alphafold/common
gunzip -c tests/test_data/sample_feats.pickle.gz > tests/test_data/sample_feats.pickle
git clone https://github.com/NVIDIA/cutlass --branch v3.6.0 --depth 1
git -C cutlass checkout bf9da7b76c766d7ee7d536afc77880a4ef1f1156
conda env config vars set CUTLASS_PATH="$PWD/cutlass"
conda env config vars set KMP_AFFINITY=none

export CC=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/gcc
export CXX=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/g++
export CUDAHOSTCXX="$CXX"
export CUTLASS_PATH="$PWD/cutlass"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

TMPDIR="$HOME/openfold-tmp" MAX_JOBS=4 python setup.py install
cd ../..
```

## checkpoint and alignments

```bash
aws s3 cp --no-sign-request --region us-east-1 \
  s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt \
  artifacts/params/finetuning_no_templ_ptm_1.pt

cp artifacts/openfold/examples/monomer/alignments/6KWC_1/mgnify_hits.sto \
  artifacts/alignments_no_templates/6KWC_1/
cp artifacts/openfold/examples/monomer/alignments/6KWC_1/uniref90_hits.sto \
  artifacts/alignments_no_templates/6KWC_1/

for pdb in 2dfb 3akq 1te1 1h1a; do
  curl -fL --retry 3 -o "artifacts/template_mmcif/${pdb}.cif" \
    "https://files.rcsb.org/download/${pdb}.cif"
done
```

`model_3_ptm` disables templates

the alignment copy has no `hhsearch_output.hhr`, so the v2.2 data pipeline creates empty template features

the mmcif directory provides files required when the openfold runner initializes

## local paths

```bash
cp .env.example .env
```

```text
PROJECT_DIR=$HOME/openfold-reproducible-inference
OPENFOLD_DIR=$PROJECT_DIR/artifacts/openfold
FASTA_DIR=$PROJECT_DIR/inputs
ALIGNMENT_DIR=$PROJECT_DIR/artifacts/alignments_no_templates
TEMPLATE_MMCIF_DIR=$PROJECT_DIR/artifacts/template_mmcif
CHECKPOINT_PATH=$PROJECT_DIR/artifacts/params/finetuning_no_templ_ptm_1.pt
OUTPUT_DIR=$PROJECT_DIR/results/run-001
CONFIG_PRESET=model_3_ptm
```

## completed environment record

`evidence/conda-explicit.txt` is the exact conda package set from the completed run

## artifact records

```bash
conda activate openfold-repro
git -C artifacts/openfold rev-parse HEAD > evidence/openfold-commit.txt
conda list --explicit > evidence/conda-explicit.txt
sha256sum inputs/6KWC_1.fasta > evidence/input-sha256.txt
sha256sum artifacts/params/finetuning_no_templ_ptm_1.pt > evidence/checkpoint-sha256.txt
sha256sum artifacts/template_mmcif/*.cif > evidence/template-mmcif-sha256.txt
```

## structure comparison

```bash
python scripts/validate_output.py \
  results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb \
  | tee evidence/validation.log

mkdir -p inputs/reference
curl -fL --retry 3 -o inputs/reference/6kwc.pdb \
  https://files.rcsb.org/download/6KWC.pdb
sha256sum inputs/reference/6kwc.pdb > evidence/reference-sha256.txt
python scripts/compare_ca_rmsd.py \
  results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb A \
  inputs/reference/6kwc.pdb A \
  | tee evidence/structure-comparison.log
```

## batch job

```bash
sbatch jobs/openfold_inference.sbatch
```

## cleanup

```bash
scancel JOB_ID
rm -rf artifacts/openfold artifacts/params artifacts/template_mmcif artifacts/alignments_no_templates
```

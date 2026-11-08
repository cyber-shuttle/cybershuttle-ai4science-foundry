#!/bin/bash
#SBATCH --job-name=openfold-msa
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=openfold-msa-%A_%a.out

# Computes alignments (MSAs and template hits) for the length-series proteins,
# one array task per FASTA file, on CPU nodes. Submit from the recipe directory:
#   sbatch slurm/compute_alignments.slurm
# Results: length-series/alignments/<ID>_<chain>/ with the same four files as
# alignments/6KWC_1/. A target whose folder already exists is skipped.

set -eo pipefail

RECIPE_DIR="${RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SERIES="$RECIPE_DIR/length-series"
DB="${AF_DATA:-/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data}"
OPENFOLD_DIR="${OPENFOLD_DIR:-$HOME/scratch/openfold}"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$HOME/scratch/.conda/envs}"
if ! command -v conda >/dev/null 2>&1; then
  module load miniforge
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENFOLD_ENV:-openfold-env}"
set -u

mapfile -t FASTAS < <(ls "$SERIES"/inputs/*.fasta)
FASTA="${FASTAS[$SLURM_ARRAY_TASK_ID]}"
TAG="$(basename "$FASTA" .fasta)"
echo "Date: $(date -Is)  Job: ${SLURM_JOB_ID}  Node: $(hostname)  Target: $TAG"

# precompute_alignments.py aligns every FASTA in its input folder, so give it
# a folder with just this target.
INPUT_DIR="$(mktemp -d)"
trap 'rm -rf "$INPUT_DIR"' EXIT
cp "$FASTA" "$INPUT_DIR/"
mkdir -p "$SERIES/alignments"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"

cd "$OPENFOLD_DIR"
python scripts/precompute_alignments.py "$INPUT_DIR" "$SERIES/alignments" \
  --uniref90_database_path "$DB/uniref90/uniref90.fasta" \
  --mgnify_database_path "$DB/mgnify/mgy_clusters_2022_05.fa" \
  --bfd_database_path "$DB/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
  --uniref30_database_path "$DB/uniref30/UniRef30_2021_03" \
  --pdb70_database_path "$DB/pdb70/pdb70" \
  --jackhmmer_binary_path "$(which jackhmmer)" \
  --hhblits_binary_path "$(which hhblits)" \
  --hhsearch_binary_path "$(which hhsearch)" \
  --cpus_per_task "$SLURM_CPUS_PER_TASK" \
  --raise_errors

ls -la "$SERIES/alignments/$TAG"
echo "Finished: $(date -Is)"

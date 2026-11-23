#!/usr/bin/env bash
# Runs the proof-of-concept: every target in poc/targets.tsv under up to three
# MSA conditions, then scores all predictions against RCSB.
#
#   example  6KWC_A only, with OpenFold's example MSA (the assignment 2 setup)
#   msa      every target with a ColabFold MMseqs2 MSA (poc/inputs/msa/)
#   single   every target with a query-only MSA (no evolutionary information)
#
# Needs: the recipe install and downloads (README steps 3 and 4) and
# poc/fetch_inputs.py. Run on a GPU node, normally through poc/poc.sbatch.
# Writes poc/results/run-<job id>/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"

POC_DIR="$RECIPE_DIR/poc"
RUN_ID="${SLURM_JOB_ID:-interactive}"
RUN_DIR="$POC_DIR/results/run-$RUN_ID"
WORK="$OPENFOLD_WORK/poc/run-$RUN_ID"
CONDITIONS="${POC_CONDITIONS:-example msa single}"

if [[ -z "${CONDA_PREFIX:-}" || "$CONDA_PREFIX" != "$OPENFOLD_ENV" ]]; then
  load_modules
  activate_env
fi

ls "$POC_DIR"/inputs/fasta/*.fasta >/dev/null 2>&1 \
  || die "no poc inputs; run first on a login node: python poc/fetch_inputs.py"

echo "== building alignment directories in $WORK"
rm -rf "$WORK"
for cond in $CONDITIONS; do mkdir -p "$WORK/fasta/$cond" "$WORK/aln/$cond"; done

for fasta in "$POC_DIR"/inputs/fasta/*.fasta; do
  id="$(basename "$fasta" .fasta)"
  seq="$(sed -n 2p "$fasta")"
  for cond in $CONDITIONS; do
    case "$cond" in
      example)
        [[ "$id" == 6KWC_A ]] || continue
        mkdir -p "$WORK/aln/example/$id"
        cp "$OPENFOLD_DIR/examples/monomer/alignments/6KWC_1/mgnify_hits.sto" \
           "$OPENFOLD_DIR/examples/monomer/alignments/6KWC_1/uniref90_hits.sto" "$WORK/aln/example/$id/"
        ;;
      msa)
        compgen -G "$POC_DIR/inputs/msa/$id/*.a3m.gz" >/dev/null || { echo "no msa for $id, skipping it in msa"; continue; }
        mkdir -p "$WORK/aln/msa/$id"
        for gz in "$POC_DIR"/inputs/msa/"$id"/*.a3m.gz; do
          gunzip -c "$gz" > "$WORK/aln/msa/$id/$(basename "$gz" .gz)"
        done
        ;;
      single)
        mkdir -p "$WORK/aln/single/$id"
        printf '>%s\n%s\n' "$id" "$seq" > "$WORK/aln/single/$id/query.a3m"
        ;;
      *) die "unknown condition: $cond" ;;
    esac
    cp "$fasta" "$WORK/fasta/$cond/"
  done
done

for cond in $CONDITIONS; do
  if ! compgen -G "$WORK/fasta/$cond/*.fasta" >/dev/null; then
    echo "== $cond: no targets, skipping"
    continue
  fi
  echo "== $cond: $(ls "$WORK/fasta/$cond" | sed 's/.fasta//' | tr '\n' ' ')"
  # run_inference.sh is the same command as the validated 6KWC run; only the
  # inputs, alignments and output folder change.
  FASTA_DIR="$WORK/fasta/$cond" ALIGNMENT_DIR="$WORK/aln/$cond" OUTPUT_DIR="$RUN_DIR/$cond" \
    bash "$RECIPE_DIR/scripts/run_inference.sh"
  rm -f "$RUN_DIR/$cond"/tmp_*.fasta
done

echo "== scoring"
python "$POC_DIR/summarize.py" "$RUN_DIR" --aln-root "$WORK/aln"
echo "poc finished: $RUN_DIR"

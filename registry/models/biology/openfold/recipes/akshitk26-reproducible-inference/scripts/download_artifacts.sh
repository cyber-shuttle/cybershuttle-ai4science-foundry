#!/usr/bin/env bash
# Downloads the model weights and inputs and checks them against the hashes
# from the validated run. Needs scripts/install_environment.sh to have run
# first (the alignments come from the pinned OpenFold clone).
# No AWS CLI or login is needed: the weights are fetched over plain HTTPS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[[ -d "$OPENFOLD_DIR/examples" ]] || die "OpenFold source not found at $OPENFOLD_DIR; run: bash scripts/install_environment.sh"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
recorded_sha() { grep -E "[/ ]$1\$" "$RECIPE_DIR/evidence/$2" | cut -d' ' -f1; }

echo "== weights: $CHECKPOINT_PATH"
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
if [[ -f "$CHECKPOINT_PATH" && "$(sha_of "$CHECKPOINT_PATH")" == "$CHECKPOINT_SHA256" ]]; then
  echo "already present, checksum ok"
else
  curl -fL --retry 3 -o "$CHECKPOINT_PATH.part" "$CHECKPOINT_URL"
  [[ "$(sha_of "$CHECKPOINT_PATH.part")" == "$CHECKPOINT_SHA256" ]] \
    || die "checkpoint checksum mismatch; delete $CHECKPOINT_PATH.part and retry"
  mv "$CHECKPOINT_PATH.part" "$CHECKPOINT_PATH"
  echo "downloaded, checksum ok"
fi

echo "== precomputed alignments (no template hits): $ALIGNMENT_DIR/6KWC_1"
mkdir -p "$ALIGNMENT_DIR/6KWC_1"
for f in mgnify_hits.sto uniref90_hits.sto; do
  cp "$OPENFOLD_DIR/examples/monomer/alignments/6KWC_1/$f" "$ALIGNMENT_DIR/6KWC_1/"
done
# hhsearch_output.hhr is left out on purpose: model_3_ptm runs template-free.
ls -l "$ALIGNMENT_DIR/6KWC_1"

echo "== template mmCIF files: $TEMPLATE_MMCIF_DIR"
# OpenFold's runner requires this directory to exist and contain mmCIF files,
# even though model_3_ptm does not use templates.
mkdir -p "$TEMPLATE_MMCIF_DIR"
for pdb in 2dfb 3akq 1te1 1h1a; do
  out="$TEMPLATE_MMCIF_DIR/$pdb.cif"
  [[ -s "$out" ]] || curl -fL --retry 3 -o "$out" "https://files.rcsb.org/download/$pdb.cif"
  if [[ "$(sha_of "$out")" != "$(recorded_sha "$pdb.cif" template-mmcif-sha256.txt)" ]]; then
    # RCSB occasionally re-issues files; this does not affect a template-free run.
    echo "note: $pdb.cif differs from the recorded checksum (RCSB file update); continuing"
  fi
done

echo "== reference structure: inputs/reference/6kwc.pdb (committed in this repo)"
[[ "$(sha_of "$RECIPE_DIR/inputs/reference/6kwc.pdb")" == "$(recorded_sha 6kwc.pdb reference-sha256.txt)" ]] \
  && echo "checksum ok" || echo "note: reference checksum differs from evidence/reference-sha256.txt"

echo "downloads finished; next: bash scripts/setup_check.sh"

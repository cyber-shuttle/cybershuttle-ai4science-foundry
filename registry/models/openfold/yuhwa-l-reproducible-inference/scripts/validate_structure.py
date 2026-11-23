#!/usr/bin/env python3
"""Validate OpenFold predictions for the proteins in proteins.tsv.

For each protein it checks that the predicted sequence matches the input
FASTA, reports pLDDT (stored in the B-factor column), compares the prediction
with the experimental chain A (TM-score, C-alpha lDDT, C-alpha RMSD; see
structure_metrics.py), writes the prediction superposed on the experimental
structure, and compares the numbers with the committed baseline in
results/measurements/measurements.csv.

Usage:
  python scripts/validate_structure.py                      # results/run/
  python scripts/validate_structure.py --results results/measurements
  python scripts/validate_structure.py 6KWC_1               # one protein
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from Bio import PDB

from structure_metrics import compare, write_superposed

RECIPE_DIR = Path(__file__).resolve().parent.parent
BASELINE = RECIPE_DIR / "results/measurements/measurements.csv"
# A reproduction passes when it is within these limits of the baseline.
TOLERANCE = {"mean_plddt": 0.5, "tm_score": 0.01, "lddt_ca": 0.01, "rmsd_ca": 0.1}


def read_fasta(path):
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))


def plddt_and_sequence(path):
    chain = next(PDB.PDBParser(QUIET=True).get_structure("p", path)[0].get_chains())
    residues = [r for r in chain if r.id[0] == " "]
    seq = "".join(PDB.Polypeptide.protein_letters_3to1.get(r.resname, "X") for r in residues)
    return np.array([r["CA"].get_bfactor() for r in residues]), seq


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", nargs="*", help="Proteins to check (default: all in proteins.tsv).")
    parser.add_argument("--results", type=Path, default=RECIPE_DIR / "results/run",
                        help="Folder with one subfolder per protein (default: results/run).")
    args = parser.parse_args()

    with open(RECIPE_DIR / "proteins.tsv") as f:
        proteins = list(csv.DictReader(f, delimiter="\t"))
    if args.targets:
        proteins = [p for p in proteins if p["target"] in args.targets]
    with open(BASELINE) as f:
        baseline = {r["target"]: r for r in csv.DictReader(f)}

    rows, all_ok = [], True
    for p in proteins:
        target = p["target"]
        run_dir = args.results / target
        prediction = run_dir / "predictions" / f"{target}_model_1_ptm_unrelaxed.pdb"
        if not prediction.exists():
            print(f"{target}: MISSING {prediction}")
            all_ok = False
            continue
        reference = RECIPE_DIR / p["reference"]
        plddt, seq = plddt_and_sequence(prediction)
        seq_ok = seq == read_fasta(RECIPE_DIR / p["fasta"])
        m = compare(prediction, reference, ref_chain="A")
        write_superposed(prediction, reference, run_dir / f"{target}_superposed_on_reference.pdb",
                         ref_chain="A")
        values = {"mean_plddt": plddt.mean(), **m}
        base = baseline[target]
        diffs_ok = all(abs(values[k] - float(base[k])) <= tol for k, tol in TOLERANCE.items())
        ok = seq_ok and diffs_ok
        all_ok &= ok
        rows.append({"target": target, "status": "ok" if ok else "FAIL",
                     "residues": len(seq), "sequence_match": seq_ok,
                     "mean_plddt": f"{values['mean_plddt']:.2f}",
                     "tm_score": f"{m['tm_score']:.3f}", "lddt_ca": f"{m['lddt_ca']:.3f}",
                     "rmsd_ca": f"{m['rmsd_ca']:.2f}", "n_matched": m["n_matched"],
                     "reference": p["reference"]})

    columns = ["target", "status", "residues", "sequence_match", "mean_plddt", "tm_score",
               "lddt_ca", "rmsd_ca", "n_matched", "reference"]
    if rows:
        args.results.mkdir(parents=True, exist_ok=True)
        with open(args.results / "validation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, columns)
            w.writeheader()
            w.writerows(rows)
    print(f"{'Protein':8} {'Residues':>8} {'Seq':>5} {'pLDDT':>6} {'TM':>6} {'lDDT':>6} {'RMSD':>6}  Result")
    for r in rows:
        b = baseline[r["target"]]
        print(f"{r['target']:8} {r['residues']:>8} {str(r['sequence_match']):>5} {r['mean_plddt']:>6} "
              f"{r['tm_score']:>6} {r['lddt_ca']:>6} {r['rmsd_ca']:>6}  {r['status']}"
              f"  (baseline {float(b['mean_plddt']):.2f} / {float(b['tm_score']):.3f} / "
              f"{float(b['lddt_ca']):.3f} / {float(b['rmsd_ca']):.2f})")
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

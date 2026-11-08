#!/usr/bin/env python3
"""Write the FASTA input and copy the experimental structure for each target.

Reads length-series/targets.tsv and uses the PDB copy on the PACE shared
filesystem, so nothing is downloaded:
  length-series/inputs/<ID>_<chain>.fasta   sequence (SEQRES) of the chain
  length-series/reference/<ID>.cif          experimental structure
"""

import argparse
import csv
import shutil
from pathlib import Path

SERIES_DIR = Path(__file__).resolve().parent.parent / "length-series"
PDB_DIR = Path("/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdb-dir", type=Path, default=PDB_DIR,
                        help="AlphaFold data folder with pdb_seqres/ and pdb_mmcif/.")
    args = parser.parse_args()

    with open(SERIES_DIR / "targets.tsv") as f:
        targets = list(csv.DictReader(f, delimiter="\t"))
    wanted = {f"{t['pdb_id'].lower()}_{t['chain']}": t for t in targets}

    sequences = {}
    with open(args.pdb_dir / "pdb_seqres/pdb_seqres.txt") as f:
        name = None
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name in wanted:
                sequences[name] = line.strip()

    (SERIES_DIR / "inputs").mkdir(exist_ok=True)
    (SERIES_DIR / "reference").mkdir(exist_ok=True)
    for key, t in wanted.items():
        seq = sequences.get(key)
        if seq is None:
            raise SystemExit(f"{key} not found in pdb_seqres.txt")
        if len(seq) != int(t["length"]):
            raise SystemExit(f"{key}: expected {t['length']} residues, found {len(seq)}")
        tag = f"{t['pdb_id']}_{t['chain']}"
        (SERIES_DIR / "inputs" / f"{tag}.fasta").write_text(f">{tag}\n{seq}\n")
        shutil.copyfile(args.pdb_dir / f"pdb_mmcif/mmcif_files/{t['pdb_id'].lower()}.cif",
                    SERIES_DIR / "reference" / f"{t['pdb_id']}.cif")
        print(f"{tag}: {len(seq)} residues")


if __name__ == "__main__":
    main()

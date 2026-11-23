#!/usr/bin/env python3
"""Check that every protein in length-series/targets.tsv meets the selection rules.

Uses the PDB copy on the PACE shared filesystem (nothing is downloaded) and the
same checks as the original search (README section 6):
  - the entry has exactly one chain, it is a protein, and has no unknown residues (X)
  - X-ray structure at 2.5 A or better
  - first released on or after 2023-01-01
  - at least 90% of residues have a C-alpha atom in the structure
  - no older PDB entry (ID starting with 1-6, deposited up to about 2020)
    contains any of about 40 evenly spaced 20-residue pieces of the sequence

Prints one line per protein and exits with status 1 if any check fails.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

SERIES_DIR = Path(__file__).resolve().parent.parent / "length-series"
PDB_DIR = Path("/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data")


def header_info(path):
    text = path.read_text()

    def field(name):
        m = re.search(rf"^{re.escape(name)}\s+(\S.*)$", text, re.M)
        return m.group(1).strip().strip("'") if m else None

    # First revision date = initial release.
    loop = text.split("_pdbx_audit_revision_history.revision_date", 1)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", loop[1]) if len(loop) == 2 else None
    release = m.group(1) if m else None
    method = field("_exptl.method")
    resolution = field("_refine.ls_d_res_high") or field("_reflns.d_resolution_high")
    n_ca = len(re.findall(r"^ATOM\s+\d+\s+C\s+CA\s", text, re.M))
    return release, method, float(resolution), n_ca


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdb-dir", type=Path, default=PDB_DIR,
                        help="AlphaFold data folder with pdb_seqres/ and pdb_mmcif/.")
    args = parser.parse_args()

    with open(SERIES_DIR / "targets.tsv") as f:
        targets = list(csv.DictReader(f, delimiter="\t"))
    ids = {t["pdb_id"].lower() for t in targets}

    chains = {pid: [] for pid in ids}  # pdb id -> [(chain, molecule type, sequence)]
    older = []                         # sequences of entries with IDs starting 1-6
    with open(args.pdb_dir / "pdb_seqres/pdb_seqres.txt") as f:
        for header in f:
            name, mol = header[1:].split()[:2]
            seq = next(f).strip()
            pid, chain = name.split("_", 1)
            if pid in chains:
                chains[pid].append((chain, mol, seq))
            elif pid[0] in "123456":
                older.append((pid, seq))

    ok = True
    for t in targets:
        pid = t["pdb_id"].lower()
        entry = chains[pid]
        chain, mol, seq = entry[0]
        release, method, resolution, n_ca = header_info(
            args.pdb_dir / f"pdb_mmcif/mmcif_files/{pid}.cif")
        pieces = {seq[i:i + 20] for i in range(0, len(seq) - 20, max(1, len(seq) // 40))}
        shared = sorted({p for p, s in older if any(k in s for k in pieces)})

        checks = {
            "one protein chain": len(entry) == 1 and mol == "mol:protein" and chain == t["chain"],
            "no unknown residues": "X" not in seq,
            "length matches targets.tsv": len(seq) == int(t["length"]),
            "X-ray": method == "X-RAY DIFFRACTION",
            "resolution <= 2.5 A": resolution <= 2.5,
            "released 2023 or later": release is not None and release >= "2023-01-01",
            "release matches targets.tsv": release == t["released"],
            ">= 90% residues resolved": n_ca / len(seq) >= 0.9,
            "no 20-residue piece in older entries": not shared,
        }
        failed = [name for name, passed in checks.items() if not passed]
        ok &= not failed
        print(f"{pid.upper()}_{chain}\t{len(seq)} residues\treleased {release}\t"
              f"{resolution} A\t{n_ca / len(seq):.0%} resolved\t"
              f"older entries sharing a piece: {len(shared)}\t"
              + ("PASS" if not failed else "FAIL: " + ", ".join(failed)), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

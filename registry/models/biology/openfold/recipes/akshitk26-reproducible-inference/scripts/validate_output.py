#!/usr/bin/env python3
"""basic validation for an openfold pdb output."""

from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: python scripts/validate_output.py PATH_TO_PDB")

pdb_path = Path(sys.argv[1])
if not pdb_path.is_file() or pdb_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty pdb: {pdb_path}")

atom_lines = []
residues = set()
plddt_values = []
with pdb_path.open() as handle:
    for line in handle:
        if line.startswith(("ATOM  ", "HETATM")):
            atom_lines.append(line)
            residues.add((line[21], line[22:26], line[26]))
            try:
                plddt_values.append(float(line[60:66]))
            except ValueError:
                pass

if not atom_lines:
    raise SystemExit("pdb has no atom or hetatm records")

print(f"pdb: {pdb_path}")
print(f"size (bytes): {pdb_path.stat().st_size}")
print(f"atom records: {len(atom_lines)}")
print(f"unique residues: {len(residues)}")
if plddt_values:
    print(f"mean pdb b-factor / predicted confidence: {sum(plddt_values) / len(plddt_values):.2f}")
print("pass: structure file with atom coordinates")

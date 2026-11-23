#!/usr/bin/env python3
"""ca rmsd after sequence alignment and rigid-body superposition."""

from pathlib import Path
import sys

import numpy as np


three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def ca_residues(path: Path, chain: str):
    residues = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            if line[12:16].strip() != "CA" or line[21].strip() != chain:
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (line[22:26], line[26])
            if key in seen:
                continue
            residue = three_to_one.get(line[17:20].strip())
            if residue is None:
                continue
            seen.add(key)
            residues.append((
                residue,
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            ))
    if not residues:
        raise ValueError(f"no ca atoms for chain {chain} in {path}")
    return residues


def align(first: str, second: str):
    rows, cols = len(first) + 1, len(second) + 1
    scores = np.zeros((rows, cols), dtype=int)
    trace = np.zeros((rows, cols), dtype=np.int8)
    scores[:, 0] = np.arange(rows) * -2
    scores[0, :] = np.arange(cols) * -2
    trace[1:, 0] = 1
    trace[0, 1:] = 2
    for i in range(1, rows):
        for j in range(1, cols):
            choices = (
                scores[i - 1, j - 1] + (2 if first[i - 1] == second[j - 1] else -1),
                scores[i - 1, j] - 2,
                scores[i, j - 1] - 2,
            )
            trace[i, j] = int(np.argmax(choices))
            scores[i, j] = choices[trace[i, j]]
    pairs = []
    i, j = len(first), len(second)
    while i or j:
        step = trace[i, j]
        if i and j and step == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (not j or step == 1):
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def rmsd_after_superposition(mobile, target):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(mobile_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    aligned = mobile_zero @ rotation + target_center
    return np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))


if len(sys.argv) != 5:
    raise SystemExit(
        "usage: python scripts/compare_ca_rmsd.py PREDICTED_PDB PREDICTED_CHAIN REFERENCE_PDB REFERENCE_CHAIN"
    )

predicted_path = Path(sys.argv[1])
predicted_chain = sys.argv[2]
reference_path = Path(sys.argv[3])
reference_chain = sys.argv[4]

predicted = ca_residues(predicted_path, predicted_chain)
reference = ca_residues(reference_path, reference_chain)
pairs = align("".join(r[0] for r in predicted), "".join(r[0] for r in reference))
matching = [(i, j) for i, j in pairs if predicted[i][0] == reference[j][0]]
if len(matching) < 3:
    raise SystemExit("fewer than three matching ca residues")

predicted_coords = np.array([predicted[i][1] for i, _ in matching])
reference_coords = np.array([reference[j][1] for _, j in matching])
identity = len(matching) / len(pairs)
rmsd = rmsd_after_superposition(predicted_coords, reference_coords)

print(f"predicted ca residues: {len(predicted)}")
print(f"reference ca residues: {len(reference)}")
print(f"aligned matching ca residues: {len(matching)}")
print(f"sequence identity across alignment: {identity:.2%}")
print(f"ca rmsd after superposition (angstrom): {rmsd:.3f}")

#!/usr/bin/env python3
"""Compare a predicted protein structure with a reference structure.

Residues are matched by a global sequence alignment, so the reference may
have missing residues or different residue numbering. All metrics use
C-alpha atoms:

  tm_score  TM-score, normalized by the number of matched reference residues.
            Uses the superposition search of the TM-score program
            (Zhang & Skolnick, Proteins 2004).
  lddt_ca   lDDT on C-alpha atoms (Mariani et al., Bioinformatics 2013):
            15 A inclusion radius, thresholds 0.5/1/2/4 A, no superposition.
  rmsd_ca   C-alpha RMSD after optimal superposition of all matched residues.

Usage:
  python structure_metrics.py PREDICTION.pdb REFERENCE.pdb [--ref-chain A]
                              [--superposed OUT.pdb]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Bio import PDB
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB.Polypeptide import protein_letters_3to1

# Selenomethionine is stored as a hetero residue in many crystal structures.
MODIFIED_RESIDUES = {"MSE": "M"}


def load_ca(path, chain_id=None):
    """Return (sequence, C-alpha coordinates) for one chain of a PDB/mmCIF file.

    Uses the first model, and the first chain if chain_id is not given.
    """
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    model = next(iter(parser.get_structure(path.stem, str(path))))
    chain = model[chain_id] if chain_id else next(iter(model))

    sequence, coords = [], []
    for residue in chain:
        name = residue.resname.upper()
        hetero = residue.id[0] != " "
        if hetero and name not in MODIFIED_RESIDUES:
            continue
        letter = MODIFIED_RESIDUES.get(name) or protein_letters_3to1.get(name)
        if letter is None or "CA" not in residue:
            continue
        sequence.append(letter)
        coords.append(residue["CA"].coord)
    if not coords:
        raise ValueError(f"No protein C-alpha atoms found in {path} chain {chain.id}")
    return "".join(sequence), np.asarray(coords, dtype=np.float64)


def match_residues(seq_a, seq_b):
    """Align two sequences; return index arrays of matched positions and identity."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.end_gap_score = 0  # unmodeled termini and tags are not penalized
    alignment = aligner.align(seq_a, seq_b)[0]

    idx_a, idx_b = [], []
    for (a0, a1), (b0, b1) in zip(*alignment.aligned):
        idx_a.extend(range(a0, a1))
        idx_b.extend(range(b0, b1))
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    identity = float(np.mean([seq_a[i] == seq_b[j] for i, j in zip(idx_a, idx_b)]))
    return idx_a, idx_b, identity


def kabsch(x, y):
    """Rotation r and translation t that best superpose x onto y (x @ r.T + t)."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    h = (x - x_mean).T @ (y - y_mean)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, y_mean - x_mean @ r.T


def rmsd(x, y):
    r, t = kabsch(x, y)
    return float(np.sqrt(np.mean(np.sum((x @ r.T + t - y) ** 2, axis=1))))


def tm_score(x, y, l_norm=None):
    """TM-score of matched coordinates x (model) and y (reference).

    Follows the TM-score program: superpose on every fragment of length
    L, L/2, L/4, ... (minimum 4), then repeatedly re-superpose on the residues
    that are close, and keep the highest score found.
    """
    n = len(x)
    l_norm = l_norm or n
    d0 = max(1.24 * np.cbrt(l_norm - 15) - 1.8, 0.5) if l_norm > 21 else 0.5
    d0_search = min(max(d0, 4.5), 8.0)

    def superpose_and_score(idx):
        r, t = kabsch(x[idx], y[idx])
        d2 = np.sum((x @ r.T + t - y) ** 2, axis=1)
        return float(np.sum(1.0 / (1.0 + d2 / d0**2)) / l_norm), d2

    def close_residues(d2, cutoff):
        # Relax the cutoff until at least 3 residues are selected.
        while True:
            idx = np.flatnonzero(d2 < cutoff**2)
            if len(idx) >= 3:
                return idx
            cutoff += 0.5

    if n < 4:
        return superpose_and_score(np.arange(n))[0]

    fragment_lengths = []
    for i in range(5):
        length = int(n / 2**i)
        if length <= 4:
            fragment_lengths.append(4)
            break
        fragment_lengths.append(length)
    else:
        fragment_lengths.append(4)

    best = 0.0
    for length in fragment_lengths:
        for start in range(n - length + 1):
            score, d2 = superpose_and_score(np.arange(start, start + length))
            best = max(best, score)
            idx = close_residues(d2, d0_search - 1)
            for _ in range(20):
                score, d2 = superpose_and_score(idx)
                best = max(best, score)
                new_idx = close_residues(d2, d0_search + 1)
                if np.array_equal(new_idx, idx):
                    break
                idx = new_idx
    return best


def lddt_ca(x, y, radius=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Global lDDT over all reference C-alpha pairs closer than `radius`."""
    dx = np.linalg.norm(x[:, None] - x[None], axis=-1)
    dy = np.linalg.norm(y[:, None] - y[None], axis=-1)
    pairs = (dy < radius) & ~np.eye(len(y), dtype=bool)
    error = np.abs(dx - dy)[pairs]
    return float(np.mean([(error < t).mean() for t in thresholds]))


def compare(prediction, reference, pred_chain=None, ref_chain=None):
    """All metrics for one prediction against one reference, as a dict."""
    pred_seq, pred_xyz = load_ca(prediction, pred_chain)
    ref_seq, ref_xyz = load_ca(reference, ref_chain)
    idx_pred, idx_ref, identity = match_residues(pred_seq, ref_seq)
    x, y = pred_xyz[idx_pred], ref_xyz[idx_ref]
    return {
        "tm_score": tm_score(x, y),
        "lddt_ca": lddt_ca(x, y),
        "rmsd_ca": rmsd(x, y),
        "n_matched": int(len(idx_ref)),
        "n_reference": int(len(ref_seq)),
        "n_prediction": int(len(pred_seq)),
        "sequence_identity": identity,
    }


def write_superposed(prediction, reference, out_path, pred_chain=None, ref_chain=None):
    """Write the prediction rotated onto the reference (fit on matched C-alpha atoms)."""
    pred_seq, pred_xyz = load_ca(prediction, pred_chain)
    ref_seq, ref_xyz = load_ca(reference, ref_chain)
    idx_pred, idx_ref, _ = match_residues(pred_seq, ref_seq)
    r, t = kabsch(pred_xyz[idx_pred], ref_xyz[idx_ref])
    structure = PDB.PDBParser(QUIET=True).get_structure("prediction", str(prediction))
    for atom in structure.get_atoms():
        atom.set_coord(atom.coord @ r.T + t)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--pred-chain", default=None)
    parser.add_argument("--ref-chain", default=None)
    parser.add_argument("--superposed", type=Path,
                        help="Also write the prediction superposed on the reference here.")
    args = parser.parse_args()
    result = compare(args.prediction, args.reference, args.pred_chain, args.ref_chain)
    if args.superposed:
        write_superposed(args.prediction, args.reference, args.superposed,
                         args.pred_chain, args.ref_chain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
from pathlib import Path
import urllib.request

import numpy as np
from Bio import PDB, SeqIO
from Bio.PDB.Polypeptide import protein_letters_3to1


BASE_DIR = Path(__file__).resolve().parent.parent
FASTA_PATH = BASE_DIR / "inputs" / "6KWC.fasta"
PREDICTION_PATH = BASE_DIR / "results" / "6KWC_1_model_1_ptm_unrelaxed.pdb"
REFERENCE_DIR = BASE_DIR / "reference"
REFERENCE_PATH = REFERENCE_DIR / "6KWC.pdb"

REFERENCE_URL = "https://files.rcsb.org/download/6KWC.pdb"


def chain_sequence(chain):
    sequence = []
    residues = []

    for residue in chain:
        if residue.id[0] != " ":
            continue

        name = residue.resname.upper()
        if name in protein_letters_3to1:
            sequence.append(protein_letters_3to1[name])
            residues.append(residue)

    return "".join(sequence), residues


def main():
    arg_parser = argparse.ArgumentParser(
        description="Validate an OpenFold 6KWC prediction."
    )
    arg_parser.add_argument(
        "--prediction",
        type=Path,
        default=PREDICTION_PATH,
        help="Predicted PDB to validate (default: included reference prediction).",
    )
    arg_parser.add_argument(
        "--aligned-output",
        type=Path,
        default=None,
        help="Where to write the superimposed prediction "
        "(default: next to the prediction).",
    )
    args = arg_parser.parse_args()
    prediction_path = args.prediction.resolve()
    aligned_path = args.aligned_output or (
        prediction_path.parent / "6KWC_1_model_1_ptm_aligned_to_6KWC.pdb"
    )

    parser = PDB.PDBParser(QUIET=True)

    # Read expected sequence.
    fasta_record = SeqIO.read(FASTA_PATH, "fasta")
    fasta_sequence = str(fasta_record.seq)

    # Read OpenFold prediction.
    predicted_structure = parser.get_structure("openfold", prediction_path)
    predicted_chain = predicted_structure[0]["A"]
    predicted_sequence, predicted_residues = chain_sequence(predicted_chain)

    # pLDDT is stored in the PDB B-factor field.
    plddt = np.array(
        [residue["CA"].get_bfactor() for residue in predicted_residues]
    )

    print("=== OpenFold prediction validation ===")
    print(f"FASTA length: {len(fasta_sequence)}")
    print(f"Prediction residues: {len(predicted_residues)}")
    print(f"Prediction sequence exact match: {predicted_sequence == fasta_sequence}")
    print()
    print(f"Mean pLDDT: {plddt.mean():.2f}")
    print(f"Median pLDDT: {np.median(plddt):.2f}")
    print(f"Min pLDDT: {plddt.min():.2f}")
    print(f"Max pLDDT: {plddt.max():.2f}")
    print()
    print("Confidence bins:")
    print(f"Very high (>=90): {(plddt >= 90).sum()}")
    print(f"Confident (70-90): {((plddt >= 70) & (plddt < 90)).sum()}")
    print(f"Low (50-70): {((plddt >= 50) & (plddt < 70)).sum()}")
    print(f"Very low (<50): {(plddt < 50).sum()}")

    # Obtain experimental structure from RCSB PDB if needed.
    REFERENCE_DIR.mkdir(exist_ok=True)

    if not REFERENCE_PATH.exists():
        print()
        print(f"Downloading experimental 6KWC structure from {REFERENCE_URL}")
        urllib.request.urlretrieve(REFERENCE_URL, REFERENCE_PATH)

    experimental_structure = parser.get_structure("experimental", REFERENCE_PATH)
    experimental_chain = experimental_structure[0]["A"]
    experimental_sequence, experimental_residues = chain_sequence(
        experimental_chain
    )

    print()
    print("=== Experimental structure comparison ===")
    print(f"Experimental residues: {len(experimental_residues)}")
    print(
        "Experimental sequence exact match: "
        f"{experimental_sequence == fasta_sequence}"
    )

    if not (
        predicted_sequence == fasta_sequence
        and experimental_sequence == fasta_sequence
    ):
        raise ValueError(
            "Sequence mismatch detected; refusing position-by-position structural alignment."
        )

    predicted_ca = [residue["CA"] for residue in predicted_residues]
    experimental_ca = [residue["CA"] for residue in experimental_residues]

    superimposer = PDB.Superimposer()
    superimposer.set_atoms(experimental_ca, predicted_ca)

    print(f"Aligned C-alpha residues: {len(predicted_ca)}")
    print(
        f"C-alpha RMSD after optimal superposition: "
        f"{superimposer.rms:.3f} Angstrom"
    )

    # Apply the optimal transformation and save the aligned prediction.
    superimposer.apply(predicted_structure.get_atoms())

    io = PDB.PDBIO()
    io.set_structure(predicted_structure)
    io.save(str(aligned_path))

    print(f"Aligned prediction written to: {aligned_path}")


if __name__ == "__main__":
    main()

# validation

## smoke test

| item | expected | observed | evidence |
|---|---|---|---|
| openfold import | module path | openfold `v2.2.0` egg | `evidence/smoke-test.log` |
| cuda | available gpu | Tesla V100-PCIE-16GB | `evidence/smoke-test.log` |
| input paths | fasta, alignments, checkpoint, mmcif directory | available | `evidence/smoke-test.log` |

## pdb output

| item | expected | observed | evidence |
|---|---|---|---|
| pdb file | non-empty `.pdb` file | 120528 bytes | `evidence/validation.log` |
| atom records | one or more | 1482 | `evidence/validation.log` |
| residue count | fasta length | 191 / 191 | `evidence/validation.log` |
| mean predicted confidence | 0 to 100 | 96.66 | `evidence/validation.log` |

## scientific check

| item | value |
|---|---|
| method | ca rmsd after sequence alignment and rigid-body superposition |
| command | `scripts/compare_ca_rmsd.py` |
| reference structure | RCSB PDB `6KWC`, chain `A` |
| aligned residues | 191 |
| sequence identity | 100.00% |
| ca rmsd | 1.154 angstrom |
| output record | `evidence/structure-comparison.log` |

the complete sequence match, mean predicted confidence of 96.66, and 1.154 angstrom ca rmsd support the predicted structure

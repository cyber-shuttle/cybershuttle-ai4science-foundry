# proof of concept: does this workflow generalize, and does pLDDT predict accuracy?

The main recipe runs OpenFold on one protein, 6KWC chain A, with an MSA that ships with OpenFold. This folder tests two questions on that same setup:

1. Does the workflow give accurate structures for other proteins of 100 to 250 residues, including a harder multi-domain one?
2. Is mean pLDDT (OpenFold's own confidence) a usable stand-in for accuracy, measured as CA RMSD and TM-score against the RCSB structure?

ICE has no sequence databases, so new targets need MSAs from somewhere. `fetch_inputs.py` gets them from the public ColabFold MMseqs2 server once, and they are committed under `inputs/msa/`. To see how much the MSA matters, every target also runs with a query-only MSA (single-sequence mode).

## targets

| id | protein | why |
|---|---|---|
| 6KWC_A | assignment 2 target, 191 residues | anchor: must reproduce pLDDT 96.66 and CA RMSD 1.154 Å |
| 3CHY_A | CheY, about 128 residues | small single domain |
| 2LZM_A | T4 lysozyme, about 164 residues | two lobes |
| 1AKE_A | adenylate kinase, about 214 residues | harder: three domains and a hinge; scored against closed 1AKE and open 4AKE |
| 8TIM_A | triosephosphate isomerase, about 247 residues | largest, TIM barrel |

Edit `targets.tsv` to change them.

## conditions

| condition | MSA | targets |
|---|---|---|
| example | OpenFold's example MSA (the assignment 2 setup) | 6KWC_A |
| msa | ColabFold MMseqs2 server, mode env, at most 10000 sequences per file | all |
| single | query sequence only | all |

Everything else is the validated command in `scripts/run_inference.sh`: OpenFold v2.2.0, `model_3_ptm`, `finetuning_no_templ_ptm_1.pt`, no templates, seed 0, no relaxation.

## commands

Run from the recipe directory, after README steps 1 to 5 (install and downloads):

```bash
# login node (needs internet), about 5 to 15 minutes
source scripts/common.sh && load_modules && activate_env
python poc/fetch_inputs.py
# GPU batch job, about 10 to 30 minutes
sbatch poc/poc.sbatch
```

If the ColabFold server cannot be reached, run `python poc/fetch_inputs.py --no-msa`. The job then runs the example and single conditions only.

## outputs

| path | contents |
|---|---|
| `inputs/fasta/`, `inputs/reference/`, `inputs/msa/` | query sequences, RCSB structures, gzipped MSAs |
| `inputs/MANIFEST.tsv`, `inputs/SHA256SUMS` | lengths, MSA depth, fetch date, checksums |
| `results/run-<job id>/<condition>/predictions/` | predicted PDBs, pLDDT in the B-factor column |
| `results/run-<job id>/summary.md`, `summary.csv` | pLDDT, CA RMSD, TM-score, identity, MSA depth, inference time |
| `results/run-<job id>/plddt_vs_rmsd.svg` | mean pLDDT against CA RMSD |
| `evidence/slurm-<job id>.out` | job log; also platform and GPU memory records |

Check: the `6KWC_A` example row must show 191/191 matched residues, pLDDT within 0.2 of 96.66 and CA RMSD within 0.05 Å of 1.154. GPU and driver change the last digits (job 7248505 gave 96.67 and 1.152). If it is further off, the setup differs from the validated run and the other rows should not be trusted.

## limits

- All targets were released before OpenFold's training cutoff, so good scores may partly be memorization. A real test needs structures released later (for example CAMEO targets).
- ColabFold MSAs are built with MMseqs2 on different databases than the jackhmmer/HHblits MSAs OpenFold was trained with.
- Five targets and one seed: the correlation is a sign check, not a measurement.
- One model (`model_3_ptm`), no templates, no relaxation.

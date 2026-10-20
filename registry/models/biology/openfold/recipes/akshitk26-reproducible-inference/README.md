# openfold monomer inference

`6KWC_1` inference on ICE

## completed run

| item | value |
|---|---|
| source | openfold `v2.2.0` at `e938c184a291bf053af3b14c1e3e8bb29aee57e2` |
| model | `model_3_ptm` with `finetuning_no_templ_ptm_1.pt` |
| input | `inputs/6KWC_1.fasta` with template-free example alignments |
| batch job | `6190401` on one RTX 6000 gpu |
| inference time | 26.21 seconds |
| peak gpu memory | 6433 MiB |
| output | `results/run-001/predictions/6KWC_1_model_3_ptm_unrelaxed.pdb` |
| validation | 191 residues, 96.66 mean confidence, 1.154 angstrom ca rmsd |

## workflow

1. `docs/CLUSTER_COMMANDS.md` covers ICE setup, artifacts, paths, batch inference, validation, and cleanup
2. `.env.example` defines the project paths
3. `jobs/openfold_inference.sbatch` defines the resource request and batch command

## records

| path | contents |
|---|---|
| `environment.yml` and `evidence/conda-explicit.txt` | environment configuration and completed conda package set |
| `inputs/MANIFEST.md` and `docs/PROVENANCE.md` | source, version, artifact, and checksum record |
| `docs/EXECUTION_RECORD.md` | batch resources, runtime, output, and failure record |
| `docs/VALIDATION.md` | smoke test and structure validation |
| `evidence/` | final job log, resource record, hashes, and validation records |

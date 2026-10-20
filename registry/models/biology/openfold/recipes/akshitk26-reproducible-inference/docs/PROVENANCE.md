# provenance

## software

| item | value | source |
|---|---|---|
| openfold repository | https://github.com/aqlaboratory/openfold | git remote |
| openfold release | `v2.2.0` | https://github.com/aqlaboratory/openfold/releases/tag/v2.2.0 |
| openfold commit | `e938c184a291bf053af3b14c1e3e8bb29aee57e2` | `evidence/openfold-commit.txt` |
| Python | 3.10 | `environment.yml` |
| PyTorch | 2.5.1 | `evidence/conda-explicit.txt` |
| CUDA runtime | 12.4 | `evidence/conda-explicit.txt` |
| gpu model | RTX 6000 | `evidence/platform-6190401.txt` |
| slurm allocation | 1 gpu, 4 cpus, 32 gb ram | `evidence/slurm-6190401.out` |
| batch job | `6190401`, completed | `evidence/slurm-6190401.out` |

## model

| item | value | source |
|---|---|---|
| configuration preset | `model_3_ptm` | `scripts/run_inference.sh` |
| checkpoint file | `finetuning_no_templ_ptm_1.pt` | openfold parameters |
| download url | `s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt` | openfold parameter bucket |
| sha-256 | `evidence/checkpoint-sha256.txt` | ICE download record |
| license | CC BY 4.0 | openfold documentation |

## input and reference

| item | value | source |
|---|---|---|
| query id | `6KWC_1` | fasta header |
| fasta sha-256 | `evidence/input-sha256.txt` | repository record |
| alignment source | openfold `examples/monomer/alignments/6KWC_1` | pinned source clone |
| alignment selection | MSA files without `hhsearch_output.hhr` | `artifacts/alignments_no_templates/6KWC_1` |
| experimental reference | `6KWC` chain `A` | RCSB PDB |
| reference download url | https://files.rcsb.org/download/6KWC.pdb | RCSB PDB |

## command record

| item | value |
|---|---|
| environment file | `environment.yml` |
| checkpoint command | `aws s3 cp --no-sign-request --region us-east-1 s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt artifacts/params/finetuning_no_templ_ptm_1.pt` |
| batch command | `sbatch jobs/openfold_inference.sbatch` |
| inference command | `bash scripts/run_inference.sh` |

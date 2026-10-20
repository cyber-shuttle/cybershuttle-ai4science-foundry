# artifact manifest

artifact source, version, checksum, and local location

| artifact | value | source | version / checksum | local path |
|---|---|---|---|---|
| openfold source | `aqlaboratory/openfold` | https://github.com/aqlaboratory/openfold | `v2.2.0`, `e938c184a291bf053af3b14c1e3e8bb29aee57e2` | `$OPENFOLD_DIR` |
| cutlass build source | `NVIDIA/cutlass` | https://github.com/NVIDIA/cutlass | `bf9da7b76c766d7ee7d536afc77880a4ef1f1156` | `$OPENFOLD_DIR/cutlass` |
| query fasta | `6KWC_1` | openfold `examples/monomer/fasta_dir/6kwc.fasta` | sha-256 in `evidence/input-sha256.txt` | `inputs/6KWC_1.fasta` |
| precomputed MSA files | `6KWC_1` | openfold `examples/monomer/alignments` | `mgnify_hits.sto`, `uniref90_hits.sto` | `$ALIGNMENT_DIR/6KWC_1` |
| template-hit file | `hhsearch_output.hhr` | openfold example alignment directory | excluded from template-free alignment copy | not used |
| checkpoint | `finetuning_no_templ_ptm_1.pt` | `s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt` | 373259620 bytes; sha-256 in `evidence/checkpoint-sha256.txt` | `$CHECKPOINT_PATH` |
| template mmcif files | `1h1a`, `1te1`, `2dfb`, `3akq` | https://files.rcsb.org/download/ | sha-256 in `evidence/template-mmcif-sha256.txt` | `$TEMPLATE_MMCIF_DIR` |
| reference structure | `6KWC` chain `A` | https://files.rcsb.org/download/6KWC.pdb | sha-256 in `evidence/reference-sha256.txt` | `inputs/reference/6kwc.pdb` |

## checksums

`evidence/` contains the ICE checksum records

```bash
sha256sum FILE
shasum -a 256 FILE
```

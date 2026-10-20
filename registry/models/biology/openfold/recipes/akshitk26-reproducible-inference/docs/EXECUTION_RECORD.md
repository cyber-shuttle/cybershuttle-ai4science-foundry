# execution record

| item | value |
|---|---|
| interactive inference | completed on Tesla V100-PCIE-16GB |
| batch job | `6190401` |
| batch state | completed, exit code `0:0` |
| batch allocation | 1 RTX 6000 gpu, 4 cpus, 32 gb ram |
| batch elapsed time | 78 seconds |
| model inference time | 26.21 seconds |
| peak gpu memory | 6433 MiB |
| output pdb | 120528 bytes, 1482 atom records, 191 residues |
| mean predicted confidence | 96.66 |
| reference comparison | 191 aligned residues, 100.00% sequence identity, 1.154 angstrom ca rmsd |

`evidence/` contains the environment record, final scheduler log, hashes, gpu memory record, and validation records

| job | state | reason or record |
|---|---|---|
| `6189984` | failed during batch setup | `set -u` stopped conda activation; `evidence/slurm-6189984.err` |
| `6190271` | failed during batch setup | scheduler spool path blocked evidence writes; `evidence/slurm-6190271.err` |
| `6190401` | completed | `evidence/slurm-6190401.out` |

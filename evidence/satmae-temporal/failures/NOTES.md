# Failures encountered and fixed

These are the assignment-required failure notes. Each one blocked a job,
was fixed in-tree, and has a matching Troubleshooting entry in
`registry/models/earth-science/satmae-temporal/README.md`.

## Job 6190908 — env build (`import timm`)

`timm==0.3.2` does `from torch._six import container_abcs`. The first
patch script imported `timm` to find site-packages, which is the broken
import. Fix: locate `timm` via `sysconfig.get_path("purelib")`. Rebuild
job 6192077 completed.

## Job 6192078 — data prepare (53,041 missing paths)

S3 val downloaded (~22 GB) but SatMAE CSV paths are `*_rgb_crop_0.jpg`
and SpaceNet only has `*_rgb.jpg`. Fix: `satmae_temporal_crop_fmow.py`
writes reconstructed AOI chips. Job 6195403: 53,041 existing, 0 missing.

```
rows=53041 existing_images=0 missing=53041
example_missing fmow/val/burial_site/burial_site_125/burial_site_125_0_rgb_crop_0.jpg
```

After crop:

```
crop_counts {'wrote': 53026, 'exists': 15, ...}
rows=53041 existing_images=53041 missing=0
```

## Job 6193340 — crop deadlock

`ProcessPoolExecutor.submit` of all 53,041 tasks filled worker pipes.
Canceled. Job 6195403 used `imap_unordered` and finished in 8m10s.

## Job 6195919 — `transforms.Scale`

This conda torchvision 0.13.1 has no `Scale`. Evaluator aliases it to
`Resize`. See `satmae_temporal_evaluate.py`.

## Job 6196065 — `torch.arange(..., dtype=np.float64)`

Naive `np.float` → `np.float64` rewrite is invalid for `torch.arange`
and also corrupted `np.float32`. Fix: `torch.float64` for the torch
call; restore `np.float32`. Checkpoint had already loaded 0 / 0 keys.

## Job 6197109 — subset companion `KeyError`

`parse_timestamp` indexes only the CSV given to the dataset. A 500-row
CSV cannot look up companion crops. Fix: `--csv val_prepared.csv`
`--eval-csv val_subset500.csv`.

#!/usr/bin/env python

import hashlib
import os
from pathlib import Path
import cebra.datasets

name = "rat-hippocampus-single-achilles"
root = Path(os.environ.get("CEBRA_DATADIR", Path(__file__).resolve().parent.parent / "data"))
root.mkdir(parents=True, exist_ok=True)
os.environ["CEBRA_DATADIR"] = str(root)
print(f"CEBRA_DATADIR = {root.resolve()}")
ds = cebra.datasets.init(name)
print(f"Loaded '{name}': neural {tuple(ds.neural.shape)}, labels {tuple(ds.continuous_index.shape)}")

for f in sorted(root.rglob("*")):
    if f.is_file():
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"{f}  {f.stat().st_size} bytes  sha256={h}")

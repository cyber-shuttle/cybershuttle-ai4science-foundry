import json
import os
import sys
import time
import urllib.request

# (chrom, 0-based start). Arbitrary picks; any region containing N is skipped.
REGIONS = [
    ("chr1", 100_000_000), ("chr2", 50_000_000), ("chr3", 120_000_000),
    ("chr5", 80_000_000), ("chr7", 100_000_000), ("chr9", 100_000_000),
    ("chr12", 80_000_000), ("chr16", 60_000_000),
]
LENGTH = 1000
MIN_REGIONS = 5
OUT_PATH = "data/hg38_regions.fasta"

kept = []
for chrom, start in REGIONS:
    url = (f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;"
           f"chrom={chrom};start={start};end={start + LENGTH}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            dna = json.load(r)["dna"].upper()
    except Exception as e:
        print(f"skip {chrom}:{start} (request failed: {e})")
        continue
    finally:
        time.sleep(2)  # UCSC asks for max of 1 request per second

    if len(dna) != LENGTH or set(dna) - set("ACGT"):
        print(f"skip {chrom}:{start} (length {len(dna)}, non-ACGT bases present)")
        continue
    kept.append((f"{chrom}:{start}-{start + LENGTH} hg38 (0-based, half-open)", dna))

if len(kept) < MIN_REGIONS:
    sys.exit(f"only {len(kept)} usable regions (need at least {MIN_REGIONS}); not writing {OUT_PATH}")

os.makedirs("data", exist_ok=True)
with open(OUT_PATH, "w") as f:
    for header, dna in kept:
        f.write(f">{header}\n{dna}\n")
print(f"wrote {len(kept)} regions to {OUT_PATH}")
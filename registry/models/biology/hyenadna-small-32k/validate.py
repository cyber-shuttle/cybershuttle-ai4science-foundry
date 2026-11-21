import json
import math
import random
import sys
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, OUT_DIR, REVISION

FASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hg38_regions.fasta")
SEED = 0


def read_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name:
        recs.append((name, "".join(buf)))
    return recs


tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, trust_remote_code=True
).to("cuda").eval()

base_ids = tokenizer.convert_tokens_to_ids(["A", "C", "G", "T"])
print("A,C,G,T token IDs:", base_ids)
to_acgt_index = {tid: i for i, tid in enumerate(base_ids)}


@torch.no_grad()
def score(seq):
    """Mean next-base loss in nats, over the full vocab and over A/C/G/T only."""
    ids = tokenizer(seq, return_tensors="pt")["input_ids"].to("cuda")
    n = len(seq)
    # Assumption from the smoke test: n bases plus one trailing special token
    assert ids.shape[1] == n + 1, f"unexpected tokenization length {ids.shape[1]} for {n} bases"
    logits = model(input_ids=ids).logits[0].float()
    pred = logits[: n - 1]            # position i predicts base i+1
    target = ids[0, 1:n]
    full = F.cross_entropy(pred, target).item()
    pred4 = pred[:, base_ids]         # renormalize over the four bases
    target4 = torch.tensor([to_acgt_index[t] for t in target.tolist()], device="cuda")
    acgt = F.cross_entropy(pred4, target4).item()
    return full, acgt


rng = random.Random(SEED)
rows = []
for name, seq in read_fasta(FASTA):
    chars = list(seq)
    rng.shuffle(chars)                # same base composition, order destroyed
    r_full, r_acgt = score(seq)
    s_full, s_acgt = score("".join(chars))
    rows.append({"region": name, "real_full": r_full, "real_acgt": r_acgt,
                 "shuffled_full": s_full, "shuffled_acgt": s_acgt})
    print(f"{name}: real={r_acgt:.4f} shuffled={s_acgt:.4f} (ACGT-only nats/base)")

mean = lambda k: sum(r[k] for r in rows) / len(rows)
ln4 = math.log(4)
wins = sum(r["real_acgt"] < r["shuffled_acgt"] for r in rows)
summary = {
    "model": MODEL, "revision": REVISION, "n_regions": len(rows), "shuffle_seed": SEED,
    "ln4_uniform_baseline": ln4,
    "mean_real_acgt": mean("real_acgt"), "mean_shuffled_acgt": mean("shuffled_acgt"),
    "mean_real_full": mean("real_full"), "mean_shuffled_full": mean("shuffled_full"),
    "regions_real_below_shuffled": wins,
}
checks = {
    "mean real loss below ln(4)": summary["mean_real_acgt"] < ln4,
    "mean real loss below mean shuffled loss": summary["mean_real_acgt"] < summary["mean_shuffled_acgt"],
    "real below shuffled in a majority of regions": wins > len(rows) / 2,
}
print(json.dumps(summary, indent=2))
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
    json.dump({"summary": summary, "checks": checks, "per_region": rows}, f, indent=2)
sys.exit(0 if all(checks.values()) else 1)
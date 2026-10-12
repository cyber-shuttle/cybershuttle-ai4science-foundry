#!/usr/bin/env python
# Checks the outputs of demo_hippocampus.py and prints PASS/FAIL with expected vs observed.
# Run: python validate.py outputs/run1     (exit code 0 = all 7 checks passed)
import argparse
import hashlib
import json
import sys
from pathlib import Path
 
import numpy as np
 
# Linear track is ~1.6 m long. Chance-level (predict the mean) median error is ~0.4 m.
# We require < 0.20 m: lenient enough for 5000 iters, far better than chance.
MAX_POS_MEDIAN_ERR_M = 0.20
CONTROL_ERR_RATIO = 0.5  # real error must be < 50% of shuffled-control error
 
results = []
 
 
def check(name, ok, expected, observed):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        expected: {expected}\n        observed: {observed}")
 
 
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    d = Path(a.run_dir)
 
    # 1. all output files exist
    needed = ["cebra_posdir.pt", "embedding_train.npy", "embedding_test.npy", "pred_test.npy",
              "labels_test.npy", "metrics.json", "run_info.json"]
    missing = [f for f in needed if not (d / f).exists() or (d / f).stat().st_size == 0]
    check("output files present", not missing, "all of " + ", ".join(needed),
          "missing: " + (", ".join(missing) if missing else "none"))
    if missing:
        sys.exit(1)
 
    info = json.load(open(d / "run_info.json"))
    metrics = json.load(open(d / "metrics.json"))
    emb_tr, emb_te = np.load(d / "embedding_train.npy"), np.load(d / "embedding_test.npy")
    dim = info["args"]["output_dimension"]
 
    # 2. embedding shapes, no NaN/Inf
    exp = f"train ({info['data']['n_train']}, {dim}), test ({info['data']['n_test']}, {dim}), all finite"
    obs = f"train {emb_tr.shape}, test {emb_te.shape}, finite={bool(np.isfinite(emb_tr).all() and np.isfinite(emb_te).all())}"
    ok = (emb_tr.shape == (info["data"]["n_train"], dim) and emb_te.shape == (info["data"]["n_test"], dim)
          and np.isfinite(emb_tr).all() and np.isfinite(emb_te).all())
    check("embedding shapes and values", ok, exp, obs)
 
    # 3-4. checkpoint hash matches, and re-running inference gives the same embedding
    ckpt = Path(info["checkpoint"]["path"])
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = d / ckpt.name
    h = sha256(ckpt)
    check("checkpoint SHA-256 matches run_info.json", h == info["checkpoint"]["sha256"],
          info["checkpoint"]["sha256"][:16] + "...", h[:16] + "...")
 
    import cebra
    import cebra.datasets
    ds = cebra.datasets.init(info["args"]["dataset"])
    X = ds.neural.numpy()
    X_te = X[info["data"]["n_train"]:]
    re_emb = cebra.CEBRA.load(str(ckpt)).transform(X_te)
    diff = float(np.max(np.abs(re_emb - emb_te)))
    check("re-running inference from checkpoint reproduces saved embedding", diff < 1e-4,
          "max |diff| < 1e-4", f"max |diff| = {diff:.2e}")
 
    # 5-7. quality vs the shuffled-label control
    m = metrics["cebra_posdir"]
    c = metrics["shuffled_control"]
    check("position decoding median error", m["pos_median_abs_err_m"] < MAX_POS_MEDIAN_ERR_M,
          f"< {MAX_POS_MEDIAN_ERR_M} m", f"{m['pos_median_abs_err_m']:.4f} m (R2={m['pos_r2']:.3f})")
    check("real model beats shuffled-label control",
          m["pos_median_abs_err_m"] < CONTROL_ERR_RATIO * c["pos_median_abs_err_m"]
          and m["pos_r2"] > c["pos_r2"],
          f"real err < {CONTROL_ERR_RATIO} x control err, and real R2 > control R2",
          f"real err {m['pos_median_abs_err_m']:.4f} m vs control {c['pos_median_abs_err_m']:.4f} m; "
          f"R2 {m['pos_r2']:.3f} vs {c['pos_r2']:.3f}")
    check("InfoNCE goodness of fit: real < shuffled",
          m["infonce_train"] is not None and c["infonce_train"] is not None
          and m["infonce_train"] < c["infonce_train"],
          "real < shuffled (notebook: ~5.53 vs ~6.16)",
          f"real {m['infonce_train']} vs shuffled {c['infonce_train']}")
 
    n_fail = results.count(False)
    print(f"\nSUMMARY: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
 
 
if __name__ == "__main__":
    main()
 

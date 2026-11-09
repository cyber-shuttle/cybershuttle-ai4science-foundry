#!/usr/bin/env python3
"""Check Clay embeddings written by clay_embed.py.

Smoke mode (--smoke; a few chips, two identical runs):
  1. shape is [N, 1024] and every value is finite
  2. determinism: the second run gives the same numbers as the first
  3. input sensitivity: different chips give clearly different embeddings

Full mode (all chips, compared with Clay's published rev2 embeddings):
  matched cosine   cos(ours_i, clay_i) for the same chip i. Pass criterion
                   (Shi's own, set in Assignment 1, not Clay's): mean >= 0.99
  control          cos(ours_i, clay_j) for a shuffled, never-matching j. Chips
                   in one tile are alike (Clay's own embeddings of different
                   chips in this tile already average ~0.90), so a high matched
                   score only means something if it is far above this.
  retrieval        for each of our embeddings, is the closest of all Clay
                   embeddings the same chip? (top-1 accuracy)
Writes <out-dir>/validation_report.json and a histogram PNG.

Usage:
  python scripts/clay_validate.py --smoke --embeddings run1.parquet \\
      --repeat run2.parquet --out-dir benchmarks/outputs/smoke
  python scripts/clay_validate.py --embeddings ours.parquet \\
      --reference m_3211002_se_12_1_20150530.parquet --out-dir .../inference
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely

EMBED_DIM = 1024
PASS_MEAN_COSINE = 0.99         # Shi's criterion from Assignment 1
DETERMINISM_MAX_ABS = 1e-5      # two identical runs on the same GPU
SENSITIVITY_MIN_REL_L2 = 0.01   # distinct chips must differ by >1% of the norm
SEED = 0


def read_embeddings(path):
    table = pq.read_table(path)
    emb = np.asarray(table.column("embeddings").to_pylist(), dtype=np.float64)
    meta = table.schema.metadata or {}
    run = json.loads(meta[b"clay_run"]) if b"clay_run" in meta else {}
    return table, emb, run


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
            "p05": float(np.percentile(v, 5)), "max": float(v.max()), "n": int(v.size)}


def check(name, passed, expected, observed):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: expected {expected}; observed {observed}")
    return {"check": name, "expected": expected, "observed": observed, "pass": bool(passed)}


def smoke(args):
    _, a, run = read_embeddings(args.embeddings)
    checks = [check("shape", a.ndim == 2 and a.shape[1] == EMBED_DIM,
                    f"[N, {EMBED_DIM}]", list(a.shape)),
              check("finite", bool(np.isfinite(a).all()), "no NaN/Inf", bool(np.isfinite(a).all()))]
    _, b, _ = read_embeddings(args.repeat)
    diff = float(np.abs(a - b).max())
    checks.append(check("determinism", diff <= DETERMINISM_MAX_ABS,
                        f"max |run1 - run2| <= {DETERMINISM_MAX_ABS}", f"{diff:.3e}"))
    norms = np.linalg.norm(a, axis=1)
    dists = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    off = dists[~np.eye(len(a), dtype=bool)]
    rel = float(off.min() / norms.mean())
    checks.append(check("input sensitivity", rel > SENSITIVITY_MIN_REL_L2 and rel > 1000 * diff / norms.mean(),
                        f"closest pair of different chips differs by > {SENSITIVITY_MIN_REL_L2:.0%} of the norm",
                        f"{rel:.2%}"))
    return {"mode": "smoke", "n_chips": len(a), "embedding_norm": summary(norms),
            "max_abs_diff_between_runs": diff, "run": run, "checks": checks}


def full(args):
    table, ours, run = read_embeddings(args.embeddings)
    ref_table = pq.read_table(args.reference)
    ref = np.asarray(ref_table.column("embeddings").to_pylist(), dtype=np.float64)
    if len(ours) != len(ref):
        raise SystemExit(f"ERROR: {len(ours)} embeddings vs {len(ref)} reference rows")

    # One-to-one match: our row i must be the same chip outline as reference row i.
    ours_geom = shapely.from_wkb(table.column("geometry").to_pylist())
    ref_geom = shapely.from_wkb(ref_table.column("geometry").to_pylist())
    gap = float(max(shapely.hausdorff_distance(p, q) for p, q in zip(ours_geom, ref_geom, strict=True)))

    u, r = unit(ours), unit(ref)
    sim = u @ r.T                                   # [ours, reference]
    matched = np.diag(sim)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(r))
    while np.any(perm == np.arange(len(r))):        # never pair a chip with itself
        perm = rng.permutation(len(r))
    mismatched = sim[np.arange(len(r)), perm]
    top1 = sim.argmax(axis=1) == np.arange(len(r))
    rank = (sim > matched[:, None]).sum(axis=1) + 1  # 1 = the true chip is the closest
    ref_ref = r @ r.T
    ref_ref_offdiag = ref_ref[~np.eye(len(r), dtype=bool)]
    rel_l2 = np.linalg.norm(ours - ref, axis=1) / np.linalg.norm(ref, axis=1)

    m = summary(matched)
    checks = [
        check("chip geometry matches reference", gap < 1e-9, "max outline gap < 1e-9 deg", f"{gap:.2e}"),
        check("finite", bool(np.isfinite(ours).all()), "no NaN/Inf", bool(np.isfinite(ours).all())),
        check("mean matched cosine (Shi's criterion)", m["mean"] >= PASS_MEAN_COSINE,
              f">= {PASS_MEAN_COSINE}", f"{m['mean']:.8f}"),
        check("matched above mismatched control", m["mean"] > float(mismatched.mean()),
              "matched mean > mismatched mean",
              f"{m['mean']:.8f} vs {float(mismatched.mean()):.4f}"),
        check("retrieval top-1", float(top1.mean()) >= 0.95,
              ">= 95% of chips find their own Clay embedding first", f"{float(top1.mean()):.1%}"),
    ]
    report = {
        "mode": "full",
        "n_chips": len(ours),
        "settings": run.get("settings"),
        "matched_cosine": m,
        "mismatched_cosine_control": summary(mismatched),
        "reference_vs_reference_different_chips": summary(ref_ref_offdiag),
        "retrieval_top1_accuracy": float(top1.mean()),
        "retrieval_rank_of_true_chip": summary(rank),
        "relative_l2_error": summary(rel_l2),
        "norm_ratio_ours_over_clay": summary(np.linalg.norm(ours, axis=1) / np.linalg.norm(ref, axis=1)),
        "worst_chips": [{"chip_index": int(i), "cosine": float(matched[i])} for i in np.argsort(matched)[:5]],
        "run": run,
        "checks": checks,
    }
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(min(mismatched.min(), matched.min(), ref_ref_offdiag.min()), 1.0, 80)
        # Counts on a log axis, so the 725 matched chips piled up near 1.0 do
        # not hide the other two groups. Clay-vs-Clay has 725*724 pairs and is
        # scaled down to the same 725 total.
        scale = np.full(len(ref_ref_offdiag), len(r) / len(ref_ref_offdiag))
        ax.hist(ref_ref_offdiag, bins=bins, weights=scale, alpha=0.4, label="Clay vs Clay, different chips (scaled)")
        ax.hist(mismatched, bins=bins, alpha=0.6, label="ours vs Clay, mismatched chips (control)")
        ax.hist(matched, bins=bins, alpha=0.8, label="ours vs Clay, same chip")
        ax.axvline(PASS_MEAN_COSINE, color="k", ls="--", lw=1, label=f"criterion {PASS_MEAN_COSINE}")
        ax.set_yscale("log")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("chips (log scale)")
        ax.set_title(f"Clay v1.5 on NAIP m_3211002_se, {len(ours)} chips\n{args.label or args.embeddings.stem}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        out = Path(args.out_dir, "validation_cosine.png")
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, type=Path, help="parquet from clay_embed.py")
    p.add_argument("--smoke", action="store_true", help="smoke checks (needs --repeat)")
    p.add_argument("--repeat", type=Path, help="second identical run, for the determinism check")
    p.add_argument("--reference", type=Path, help="Clay reference parquet (full mode)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--label", default="", help="name of this comparison, shown in the report and plot")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    args = p.parse_args()
    if args.smoke and not args.repeat:
        p.error("--smoke needs --repeat")
    if not args.smoke and not args.reference:
        p.error("full mode needs --reference")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = smoke(args) if args.smoke else full(args)
    report = {"label": args.label, "embeddings": str(args.embeddings),
              "reference": str(args.reference) if args.reference else None, **report}
    report["result"] = "PASS" if all(c["pass"] for c in report["checks"]) else "FAIL"
    out = args.out_dir / "validation_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"VALIDATION {report['result']}")
    sys.exit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()

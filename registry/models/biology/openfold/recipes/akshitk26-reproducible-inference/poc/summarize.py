#!/usr/bin/env python3
"""Score every proof-of-concept prediction against its RCSB structure.

Writes into the run directory:
  summary.csv            one row per prediction and reference
  summary.md             the same table, MSA vs single-sequence deltas, correlations
  plddt_vs_rmsd.svg      mean pLDDT against CA RMSD, one point per prediction

usage: python poc/summarize.py poc/results/run-<job id> [--aln-root DIR]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POC_DIR.parent / "scripts"))
from compare_ca_rmsd import align, ca_residues, rmsd_after_superposition  # noqa: E402

PRESET = "model_3_ptm"
CONDITION_LABELS = {
    "example": "OpenFold example MSA",
    "msa": "ColabFold MSA",
    "single": "single sequence",
}
COLORS = {"example": "#1f77b4", "msa": "#2ca02c", "single": "#d62728"}


def read_targets():
    targets = {}
    for line in (POC_DIR / "targets.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tid, pdb, chain, extra = (line.split("\t") + ["-"] * 4)[:4]
        refs = [(pdb, chain)]
        if extra.strip() not in ("", "-"):
            refs += [tuple(x.split(":")) for x in extra.split(",")]
        targets[tid] = refs
    return targets


def kabsch(mobile, target):
    """Rotation and translation that move mobile onto target (least squares)."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, tc - mc @ rot


def tm_score(mobile, target, length):
    """TM-score for a fixed residue pairing, normalised by the reference length.

    Uses the TM-score search heuristic (fragment seeds, then iterative
    refinement on close residues). It can only find superpositions, so the value
    is a lower bound of the exact TM-score and is usually within 0.01 of it.
    """
    n = len(mobile)
    d0 = max(0.5, 1.24 * (max(length, 19) - 15) ** (1.0 / 3.0) - 1.8)
    d_search = min(max(d0, 4.5), 8.0)
    best = 0.0
    frag = n
    while frag >= 4:
        step = max(1, frag // 2)
        for start in range(0, n - frag + 1, step):
            idx = np.arange(start, start + frag)
            for _ in range(20):
                rot, shift = kabsch(mobile[idx], target[idx])
                dist = np.linalg.norm(mobile @ rot + shift - target, axis=1)
                best = max(best, float(np.sum(1.0 / (1.0 + (dist / d0) ** 2)) / length))
                cut = d_search
                new = np.where(dist < cut)[0]
                while len(new) < 3 and cut < 20:
                    cut += 0.5
                    new = np.where(dist < cut)[0]
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
        frag //= 2
    return best


def score(pred_path, ref_path, ref_chain):
    predicted = ca_residues(pred_path, "A")
    reference = ca_residues(ref_path, ref_chain)
    pairs = align("".join(r[0] for r in predicted), "".join(r[0] for r in reference))
    matching = [(i, j) for i, j in pairs if predicted[i][0] == reference[j][0]]
    p = np.array([predicted[i][1] for i, _ in matching])
    r = np.array([reference[j][1] for _, j in matching])
    return {
        "pred_residues": len(predicted),
        "ref_residues": len(reference),
        "matched": len(matching),
        "identity": len(matching) / len(pairs),
        "ca_rmsd": float(rmsd_after_superposition(p, r)),
        "tm_score": tm_score(p, r, len(reference)),
    }


def mean_plddt(pred_path):
    atoms, ca = [], []
    for line in pred_path.read_text().splitlines():
        if line.startswith("ATOM  "):
            value = float(line[60:66])
            atoms.append(value)
            if line[12:16].strip() == "CA":
                ca.append(value)
    return float(np.mean(atoms)), ca


def msa_depth(aln_dir):
    """Sequences in the MSA files of one target (before OpenFold removes duplicates)."""
    if aln_dir is None or not aln_dir.is_dir():
        return ""
    total = 0
    for f in aln_dir.iterdir():
        if f.suffix == ".a3m":
            total += sum(1 for line in f.open() if line.startswith(">"))
        elif f.suffix == ".sto":
            names = {line.split()[0] for line in f.open()
                     if line.strip() and not line.startswith(("#", "//"))}
            total += len(names)
    return total


def rank(values):
    order = np.argsort(values)
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def correlations(rows):
    x = np.array([r["mean_plddt"] for r in rows])
    y = np.array([r["ca_rmsd"] for r in rows])
    if len(rows) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rank(x), rank(y))[0, 1])


def svg_plot(rows, path):
    w, h, left, right, top, bottom = 640, 440, 70, 190, 30, 60
    xs = [r["mean_plddt"] for r in rows]
    ys = [r["ca_rmsd"] for r in rows]
    x0, x1 = min(30.0, min(xs) - 2), 100.0
    y0, y1 = 0.0, max(4.0, max(ys) * 1.1)
    px = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)
    py = lambda v: h - bottom - (v - y0) / (y1 - y0) * (h - top - bottom)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="black"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="black"/>']
    for t in range(int(x0 // 10 * 10), 101, 10):
        if t >= x0:
            out.append(f'<text x="{px(t):.1f}" y="{h-bottom+18}" text-anchor="middle">{t}</text>')
    step = 1.0 if y1 <= 8 else 2.0 if y1 <= 16 else 5.0
    t = 0.0
    while t <= y1:
        out.append(f'<text x="{left-8}" y="{py(t)+4:.1f}" text-anchor="end">{t:g}</text>')
        out.append(f'<line x1="{left}" y1="{py(t):.1f}" x2="{w-right}" y2="{py(t):.1f}" stroke="#ddd"/>')
        t += step
    out.append(f'<line x1="{px(70):.1f}" y1="{top}" x2="{px(70):.1f}" y2="{h-bottom}" stroke="#999" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{(left+w-right)/2}" y="{h-15}" text-anchor="middle">mean pLDDT</text>')
    out.append(f'<text x="18" y="{(top+h-bottom)/2}" text-anchor="middle" transform="rotate(-90 18 {(top+h-bottom)/2})">CA RMSD to RCSB (Å)</text>')
    for r in rows:
        c = COLORS.get(r["condition"], "black")
        out.append(f'<circle cx="{px(r["mean_plddt"]):.1f}" cy="{py(r["ca_rmsd"]):.1f}" r="5" fill="{c}"/>')
        out.append(f'<text x="{px(r["mean_plddt"])+7:.1f}" y="{py(r["ca_rmsd"])-6:.1f}" font-size="10">{r["label"]}</text>')
    for k, (cond, label) in enumerate(CONDITION_LABELS.items()):
        y = top + 10 + 20 * k
        out.append(f'<circle cx="{w-right+20}" cy="{y}" r="5" fill="{COLORS[cond]}"/>')
        out.append(f'<text x="{w-right+30}" y="{y+4}">{label}</text>')
    out.append(f'<text x="{w-right+12}" y="{top+80}" font-size="10" fill="#555">dashed line: pLDDT 70</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--aln-root", type=Path, default=None,
                        help="alignment directories used by run_poc.sh, to report MSA depth")
    args = parser.parse_args()

    targets = read_targets()
    ref_dir = POC_DIR / "inputs" / "reference"
    rows = []
    for cond in CONDITION_LABELS:
        pred_dir = args.run_dir / cond / "predictions"
        if not pred_dir.is_dir():
            continue
        timings_file = args.run_dir / cond / "timings.json"
        timings = json.loads(timings_file.read_text()) if timings_file.is_file() else {}
        for pred in sorted(pred_dir.glob(f"*_{PRESET}_unrelaxed.pdb")):
            tid = pred.name[: -len(f"_{PRESET}_unrelaxed.pdb")]
            plddt, ca_plddt = mean_plddt(pred)
            depth = msa_depth(args.aln_root / cond / tid if args.aln_root else None)
            for k, (pdb, chain) in enumerate(targets.get(tid, [])):
                s = score(pred, ref_dir / f"{pdb.lower()}.pdb", chain)
                rows.append({
                    "target": tid, "condition": cond, "reference": f"{pdb}:{chain}",
                    "primary_reference": k == 0, "msa_depth": depth,
                    "mean_plddt": round(plddt, 2),
                    "frac_ca_plddt_over_70": round(float(np.mean(np.array(ca_plddt) > 70)), 3),
                    "ca_rmsd": round(s["ca_rmsd"], 3), "tm_score": round(s["tm_score"], 3),
                    "identity": round(s["identity"], 4), "matched": s["matched"],
                    "pred_residues": s["pred_residues"], "ref_residues": s["ref_residues"],
                    "inference_s": round(timings.get(tid, {}).get("inference", float("nan")), 1),
                    "label": f"{tid}" if k == 0 else f"{tid} vs {pdb}",
                })
    if not rows:
        raise SystemExit(f"no predictions found under {args.run_dir}")

    fields = [f for f in rows[0] if f != "label"]
    with (args.run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md = [f"# proof-of-concept summary: {args.run_dir.name}", "",
          "| target | condition | reference | MSA seqs | mean pLDDT | CA pLDDT>70 | CA RMSD (Å) | TM-score | identity | matched CA | inference (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['target']} | {CONDITION_LABELS[r['condition']]} | {r['reference']} | {r['msa_depth']} | "
                  f"{r['mean_plddt']:.2f} | {r['frac_ca_plddt_over_70']:.0%} | {r['ca_rmsd']:.3f} | {r['tm_score']:.3f} | "
                  f"{r['identity']:.2%} | {r['matched']}/{r['ref_residues']} | {r['inference_s']} |")

    primary = [r for r in rows if r["primary_reference"]]
    by_key = {(r["target"], r["condition"]): r for r in primary}
    md += ["", "## MSA vs single sequence", "",
           "| target | pLDDT with MSA | pLDDT single | RMSD with MSA (Å) | RMSD single (Å) |", "|---|---|---|---|---|"]
    for tid in targets:
        a, b = by_key.get((tid, "msa")), by_key.get((tid, "single"))
        if a and b:
            md.append(f"| {tid} | {a['mean_plddt']:.2f} | {b['mean_plddt']:.2f} | {a['ca_rmsd']:.3f} | {b['ca_rmsd']:.3f} |")

    md += ["", "## does mean pLDDT track accuracy?", ""]
    corr = correlations(primary)
    if corr:
        md.append(f"Over {len(primary)} predictions (primary reference only): Pearson r(pLDDT, CA RMSD) = {corr[0]:.2f}, "
                  f"Spearman rho = {corr[1]:.2f}. Negative means higher confidence goes with lower error. "
                  f"With this few points, treat the sign and rough size as the result, not the exact value.")
    else:
        md.append("Not enough predictions to compute a correlation.")
    md += ["", "TM-score is computed for the fixed sequence pairing with the TM-score search heuristic, "
           "so it is a lower bound of the exact value. pLDDT is the atom average, the same number "
           "`scripts/validate_output.py` prints."]
    (args.run_dir / "summary.md").write_text("\n".join(md) + "\n")
    svg_plot(primary, args.run_dir / "plddt_vs_rmsd.svg")
    print("\n".join(md))
    print(f"\nwrote {args.run_dir}/summary.csv, summary.md, plddt_vs_rmsd.svg")


if __name__ == "__main__":
    main()

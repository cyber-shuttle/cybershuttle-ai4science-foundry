#!/usr/bin/env python
"""Score predictions vs experimental structures; check success criteria.

- Metrics as in train_openfold.py::_compute_validation_metrics: lDDT-Ca, Ca-RMSD, GDT-TS
- Writes results/metrics.csv (per prediction) and results/summary.json
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from openfold.data import mmcif_parsing
from openfold.np import protein, residue_constants
from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import gdt_ts

torch.set_num_threads(1)  # login node: 1-CPU quota, 64 visible cores
DATA, RUNS, REPO = (Path(os.environ[k]) for k in ("OF_DATA", "OF_RUNS", "OF_REPO"))
RESULTS = Path(__file__).resolve().parents[1] / "results"
CA = residue_constants.atom_order["CA"]
# Published validation lDDT-Ca (OpenFold_Parameters.md)
PUBLISHED = {"openfold_templates": 0.9097, "openfold_no_templates": 0.9025}
PAIRS = {
    "templates": ("openfold_templates", "alphafold2_templates"),
    "no_templates": ("openfold_no_templates", "alphafold2_no_templates"),
}
SMOKE_REFERENCE = REPO / "examples/monomer/sample_predictions/6KWC_1_model_1_ptm_unrelaxed.pdb"


def ca_metrics(pred_pos, gt_pos, gt_mask):
    pred_pos, gt_pos, gt_mask = (torch.as_tensor(x, dtype=torch.float32) for x in (pred_pos, gt_pos, gt_mask))
    mask_ca = gt_mask[..., CA]
    gt_ca = (gt_pos * gt_mask[..., None])[..., CA, :]
    pred_ca = (pred_pos * gt_mask[..., None])[..., CA, :]
    superimposed, rmsd = superimpose(gt_ca, pred_ca, mask_ca)
    return {
        "lddt_ca": float(lddt_ca(pred_pos, gt_pos, gt_mask, eps=1e-8, per_residue=False)),  # globals.eps
        "rmsd_ca": float(rmsd),
        "gdt_ts": float(gdt_ts(superimposed, gt_ca, mask_ca)),
    }


def score_predictions():
    rows = []
    manifest = csv.DictReader(open(RESULTS / "data_manifest.tsv"), delimiter="\t")
    for t in (r for r in manifest if r["status"] == "ok"):
        cif = (DATA / "mmcif" / f"{t['pdb']}.cif").read_text()
        mmcif = mmcif_parsing.parse(file_id=t["pdb"], mmcif_string=cif).mmcif_object
        gt_pos, gt_mask = mmcif_parsing.get_atom_coords(mmcif, t["chain"])
        aatype = [residue_constants.restype_order.get(a, 20) for a in mmcif.chain_to_seqres[t["chain"]]]
        for run in sorted(p for p in (RUNS / t["set"]).iterdir() if p.is_dir()):
            pdb = next(run.glob(f"predictions/{t['tag']}_*_unrelaxed.pdb"), None)
            row = {"set": t["set"], "config": run.name, "tag": t["tag"], "length": int(t["length"])}
            if pdb is None:
                rows.append(row)
                continue
            pred = protein.from_pdb_string(pdb.read_text())
            assert list(pred.aatype) == aatype, f"{pdb} sequence differs from SEQRES"
            timings = json.loads((run / "timings.json").read_text())
            rows.append({**row, **ca_metrics(pred.atom_positions, gt_pos, gt_mask),
                         "plddt": float(pred.b_factors[:, CA].mean()),
                         "inference_s": timings[t["tag"]]["inference"]})
    return rows


def bootstrap_ci(x, n=10000, seed=0):
    means = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def peak_gpu_mib(run):
    lines = (run / "gpu_usage.csv").read_text().splitlines()[1:]
    return max(int(line.split(",")[1].split()[0]) for line in lines)


def summarize(rows):
    done = [r for r in rows if "lddt_ca" in r]
    by = lambda s, c: {r["tag"]: r for r in done if r["set"] == s and r["config"] == c}
    configs = sorted({r["config"] for r in rows if r["set"] == "cameo"})
    summary = {"cameo": {}, "paired_openfold_minus_alphafold2": {}, "smoke_6KWC": {}}

    for c in configs:
        res = by("cameo", c)
        lddt = np.array([r["lddt_ca"] for r in res.values()])
        plddt = np.array([r["plddt"] for r in res.values()])
        s = {
            "n_scored": len(res),
            "n_missing": sum(r["set"] == "cameo" and r["config"] == c and "lddt_ca" not in r for r in rows),
            "mean_lddt_ca": float(lddt.mean()),
            "mean_gdt_ts": float(np.mean([r["gdt_ts"] for r in res.values()])),
            "median_rmsd_ca": float(np.median([r["rmsd_ca"] for r in res.values()])),
            "mean_plddt": float(plddt.mean()),
            "pearson_plddt_vs_lddt_ca": float(np.corrcoef(plddt, lddt)[0, 1]),
            "inference_gpu_hours": sum(r["inference_s"] for r in res.values()) / 3600,
            # allocator-reserved memory; bounded by the card, not a requirement
            "nvidia_smi_peak_memory_mib": peak_gpu_mib(RUNS / "cameo" / c),
        }
        if c in PUBLISHED:
            delta = s["mean_lddt_ca"] - PUBLISHED[c]
            s.update(published_mean_lddt_ca=PUBLISHED[c], delta_vs_published=delta,
                     criterion_within_0_02_of_published=bool(abs(delta) <= 0.02))
        summary["cameo"][c] = s

    for mode, (of, af) in PAIRS.items():
        a, b = by("cameo", of), by("cameo", af)
        tags = sorted(a.keys() & b.keys())
        diff = np.array([a[t]["lddt_ca"] - b[t]["lddt_ca"] for t in tags])
        summary["paired_openfold_minus_alphafold2"][mode] = {
            "n": len(tags),
            "mean_diff_lddt_ca": float(diff.mean()),
            "bootstrap_95ci": bootstrap_ci(diff),
            "fraction_openfold_at_least_as_good": float((diff >= 0).mean()),
            "criterion_abs_mean_diff_le_0_01": bool(abs(diff.mean()) <= 0.01),
        }

    # Bundled 6KWC reference vs both model_1_ptm weight sets
    ref = protein.from_pdb_string(SMOKE_REFERENCE.read_text())
    smoke = {"bundled_reference_mean_plddt": float(ref.b_factors[:, CA].mean())}
    for c in PAIRS["templates"]:
        ours = protein.from_pdb_string(next((RUNS / "smoke" / c / "predictions").glob("6KWC_1_*")).read_text())
        _, rmsd = superimpose(torch.tensor(ref.atom_positions[:, CA]), torch.tensor(ours.atom_positions[:, CA]),
                              torch.ones(len(ref.aatype)))
        smoke[c] = {"ca_rmsd_vs_bundled_reference_A": float(rmsd), "mean_plddt": float(ours.b_factors[:, CA].mean())}
    smoke["criterion_rmsd_le_0_5A"] = all(smoke[c]["ca_rmsd_vs_bundled_reference_A"] <= 0.5 for c in PAIRS["templates"])
    smoke["lddt_ca_vs_experimental"] = {c: r["lddt_ca"] for c in configs for r in by("smoke", c).values()}
    summary["smoke_6KWC"] = smoke
    return summary


def main():
    rows = score_predictions()
    cols = ["set", "config", "tag", "length", "lddt_ca", "gdt_ts", "rmsd_ca", "plddt", "inference_s"]
    with open(RESULTS / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, cols, restval="")
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    summary = summarize(rows)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

# Render the predicted structures of an inference or measurement run with PyMOL.
# Run from the recipe directory:
#   pymol -cq scripts/render_structures.py -- RESULTS_DIR IMAGE_DIR
#
# For each target folder it writes two images:
#   <target>_plddt.png                     prediction colored by pLDDT
#   <target>_openfold_vs_experimental.png  prediction (blue) on the experimental chain (gray)

import csv
import json
import sys
from pathlib import Path

from pymol import cmd

results = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

# A measurement run (section 6) has measurements.csv; a plain inference run
# has validation.csv, written by scripts/validate_structure.py (section 4).
table = results / "measurements.csv"
if not table.exists():
    table = results / "validation.csv"
with open(table, newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["status"] == "ok"]

for row in rows:
    target = row["target"]
    run_dir = results / target
    if (run_dir / "result.json").exists():
        result = json.loads((run_dir / "result.json").read_text())
        chain, reference = result["reference_chain"], Path(result["reference"])
    else:
        chain, reference = "A", Path(row["reference"])
    if not reference.exists():  # results made in another user's checkout
        reference = next(p for d in ("reference", "length-series/reference")
                         if (p := Path(d) / reference.name).exists())

    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)

    # Prediction colored by pLDDT (stored in the B-factor column).
    cmd.load(str(run_dir / f"{target}_superposed_on_reference.pdb"), "openfold")
    cmd.hide("everything")
    cmd.show("cartoon", "openfold")
    cmd.spectrum("b", "red_yellow_green", "openfold", minimum=50, maximum=100)
    cmd.orient("openfold")
    cmd.ray(1200, 900)
    cmd.png(str(out_dir / f"{target}_plddt.png"))

    # Prediction on the experimental chain, both in the reference frame.
    cmd.load(str(reference), "experimental")
    cmd.remove(f"experimental and not chain {chain}")
    cmd.remove("experimental and not polymer.protein")
    cmd.hide("everything")
    cmd.show("cartoon", "openfold or experimental")
    cmd.color("marine", "openfold")
    cmd.color("gray70", "experimental")
    cmd.orient("openfold or experimental")
    cmd.ray(1200, 900)
    cmd.png(str(out_dir / f"{target}_openfold_vs_experimental.png"))
    print(f"{target}: images written to {out_dir}")

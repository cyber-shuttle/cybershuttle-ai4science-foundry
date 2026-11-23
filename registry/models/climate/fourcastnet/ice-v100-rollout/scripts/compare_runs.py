#!/usr/bin/env python3
"""Repeatability check: compare two z500_skill_<jobid>.csv files from separate runs.

    python scripts/compare_runs.py outputs/z500_skill_A.csv outputs/z500_skill_B.csv
"""
import csv
import sys


def load(path):
    with open(path) as f:
        return [(float(r["lead_hours"]), float(r["z500_acc"]), float(r["z500_rmse"])) for r in csv.DictReader(f)]


a, b = load(sys.argv[1]), load(sys.argv[2])
if len(a) != len(b):
    sys.exit(f"different lengths: {len(a)} vs {len(b)}")
dacc = max(abs(x[1] - y[1]) for x, y in zip(a, b))
drmse = max(abs(x[2] - y[2]) for x, y in zip(a, b))
print(f"steps compared: {len(a)}")
print(f"max |ACC difference|:  {dacc:.3e}")
print(f"max |RMSE difference|: {drmse:.3e}")

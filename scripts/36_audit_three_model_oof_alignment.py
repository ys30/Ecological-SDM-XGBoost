#!/usr/bin/env python3
"""Audit record alignment of the three published spatial CV prediction files."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
def key(row):
    return tuple(row[field] for field in ("id", "plot", "fold", "observed"))
def indexed(rows):
    output = {}
    for row in rows:
        k = key(row)
        assert k not in output, ("duplicate record", k)
        output[k] = row
    return output

bayes = indexed(read("bayes_spatial_cv_predictions.csv"))
xgb = indexed(read("spatial_2d_block_cv_predictions.csv"))
maxent = indexed(read("maxent_spatial_2d_predictions.csv"))
assert len(bayes) == 9336 and len(xgb) == 9336
assert set(bayes) == set(xgb), "Bayesian/XGBoost held-out keys differ"
assert set(maxent).issubset(bayes), "MaxEnt includes unexpected held-out records"
for k, row in maxent.items():
    assert row["species"] == bayes[k]["species"], ("species mismatch", k)
print("PASS: Bayesian and XGBoost have 9,336 identical held-out keys")
print(f"PASS: {len(maxent)} MaxEnt rows align with Bayesian species and held-out keys")

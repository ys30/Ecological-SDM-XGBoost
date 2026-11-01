#!/usr/bin/env python3
"""Verify 15 three-model plotting rows exactly match the species benchmark CSV."""
import csv
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
def load(filename):
    with (root / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
species = load("matched_bayesian_maxent_xgboost_species.csv")
plot = load("matched_three_model_plot_data.csv")
models = ("Bayesian", "MaxEnt", "XGBoost")
assert len(species) == 5
assert len(plot) == 15
assert sum(int(row["folds"]) for row in species) == 23
reference = {row["species"]: row for row in species}
seen = set()
for row in plot:
    key = (row["species"], row["model"])
    assert key not in seen, ("duplicate plot row", key)
    seen.add(key)
    assert row["species"] in reference and row["model"] in models
    src = reference[row["species"]]
    for metric in ("AUC", "AP"):
        expected = float(src[f'{row["model"].lower()}_{metric.lower()}'])
        actual = float(row[metric])
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (key, metric)
assert seen == {(s, m) for s in reference for m in models}
print("PASS: all 15 figure-source rows and 30 AUC/AP values match the verified species benchmark")

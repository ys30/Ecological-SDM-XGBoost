#!/usr/bin/env python3
"""Validate archived five-fold Bayesian long-chain metric reconciliation."""
import csv
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5, (name, len(rows))
    folds = [int(r["fold"]) for r in rows]
    assert sorted(folds) == [1, 2, 3, 4, 5], (name, folds)
    return {int(r["fold"]): r for r in rows}


original = load("bayes_spatial_cv_metrics.csv")
confirmed = load("bayesian_five_fold_confirmed_metrics.csv")
assert sum(int(v["n"]) for v in original.values()) == 9336
assert sum(int(v["n"]) for v in confirmed.values()) == 9336
max_changes = {"auc": 0.0, "ap": 0.0, "logloss": 0.0}
for fold in range(1, 6):
    a, b = original[fold], confirmed[fold]
    assert int(a["n"]) == int(b["n"]), ("fold sample mismatch", fold)
    for metric in max_changes:
        x, y = float(a[metric]), float(b[metric])
        assert math.isfinite(x) and math.isfinite(y), (fold, metric)
        assert 0 <= x <= 1 and 0 <= y <= 1, (fold, metric)
        delta = abs(y - x)
        max_changes[metric] = max(max_changes[metric], delta)
        print(f"fold={fold} metric={metric} original={x:.9f} confirmed={y:.9f} abs_delta={delta:.9f}")
assert max_changes["auc"] < 0.00033, max_changes
assert max_changes["ap"] < 0.0014, max_changes
assert max_changes["logloss"] < 0.0002, max_changes
print(f"PASS: five-fold long-chain metric stability; max absolute changes {max_changes}")
print("LIMITATION: these are fold-level metrics; confirmed per-record OOF predictions were not archived.")

#!/usr/bin/env python3
"""Preflight historical benchmark counts against current spatial fold assignments."""
import csv
from collections import Counter
from pathlib import Path

def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

data = read("data/train.csv")
folds = read("results/spatial_2d_block_assignments.csv")
archive = read("results/matched_bayesian_maxent_xgboost_folds.csv")
plot_to_fold = {row["plot"]: row["fold"] for row in folds}
assert len(plot_to_fold) == len(folds), "Duplicate plot in fold map"
assert all(row["plot"] in plot_to_fold for row in data), "Missing plot fold"
counts = Counter()
positive = Counter()
for row in data:
    key = (row["Species"], plot_to_fold[row["plot"]])
    counts[key] += 1
    positive[key] += int(row["pres.abs"])
assert len(data) == 9336 and len(folds) == 1167
assert len(archive) == 23
for row in archive:
    key = (row["species"], row["fold"])
    assert counts[key] == int(row["n"]), f"Sample mismatch: {key}"
    assert positive[key] == int(row["positives"]), f"Positive mismatch: {key}"
print(f"PASS: {len(data)} observations, {len(folds)} plots, "
      f"{len(archive)} archived cells, 0 sample/positive mismatches")

#!/usr/bin/env python3
"""Audit exact record, taxon, fold and score alignment across three OOF files."""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
KEY = ("id", "plot", "fold", "observed")


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def indexed(rows, label):
    output = {}
    ids = set()
    for row in rows:
        k = tuple(row[field] for field in KEY)
        assert k not in output, (label, "duplicate key", k)
        assert row["id"] not in ids, (label, "duplicate id", row["id"])
        assert row["observed"] in ("0", "1"), (label, "invalid outcome", k)
        assert row["fold"] in ("1", "2", "3", "4", "5"), (label, "invalid fold", k)
        ids.add(row["id"])
        output[k] = row
    return output


bayes = indexed(read("bayes_spatial_cv_predictions.csv"), "Bayesian")
xgb = indexed(read("spatial_2d_block_cv_predictions.csv"), "XGBoost")
maxent = indexed(read("maxent_spatial_2d_predictions.csv"), "MaxEnt")
assert len(bayes) == len(xgb) == 9336
assert len(maxent) == 5373
assert set(bayes) == set(xgb), "Bayesian/XGBoost held-out keys differ"
assert set(maxent).issubset(bayes), "MaxEnt contains unexpected held-out records"

plot_folds = defaultdict(set)
species_folds = set()
for k, row in bayes.items():
    plot_folds[row["plot"]].add(row["fold"])
    species_folds.add((row["species"], row["fold"]))
    assert row["species"].startswith("nsw"), ("unexpected species label", k)
    for field in ("mean_probability", "lower_95", "upper_95"):
        score = float(row[field])
        assert math.isfinite(score) and 0 <= score <= 1, (field, k, score)
    assert float(row["lower_95"]) <= float(row["upper_95"]), ("interval order", k)
    score = float(xgb[k]["predicted_probability"])
    assert math.isfinite(score) and 0 <= score <= 1, ("XGBoost score", k)

assert len(plot_folds) == 1167, ("plot count", len(plot_folds))
assert all(len(folds) == 1 for folds in plot_folds.values()), "plot spans multiple folds"
assert len(species_folds) == 40, ("taxon-fold count", len(species_folds))
matched_groups = set()
for k, row in maxent.items():
    assert row["species"] == bayes[k]["species"], ("species mismatch", k)
    score = float(row["relative_suitability"])
    assert math.isfinite(score) and 0 <= score <= 1, ("MaxEnt score", k)
    matched_groups.add((row["species"], row["fold"]))
assert len(matched_groups) == 23, ("matched taxon-fold groups", len(matched_groups))

print("PASS: Bayesian/XGBoost 9,336 identical held-out records; MaxEnt 5,373 aligned")
print("PASS: 1,167 plots stay within one fold; 40 taxon-fold groups, 23 matched")
print("PASS: unique IDs, observed labels, anonymous species, finite bounded scores")

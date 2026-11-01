#!/usr/bin/env python3
"""Recalculate 23 matched species-fold ROC AUC and AP metrics from OOF scores."""
import csv
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results"
def load(filename):
    with (BASE / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
def key(row):
    return row["id"], row["plot"], row["fold"], row["observed"]
def make_index(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise AssertionError(("duplicate", k))
        result[k] = row
    return result

bayes = make_index(load("bayes_spatial_cv_predictions.csv"))
xgb = make_index(load("spatial_2d_block_cv_predictions.csv"))
maxent = make_index(load("maxent_spatial_2d_predictions.csv"))
matched = load("matched_bayesian_maxent_xgboost_folds.csv")
assert len(bayes) == len(xgb) == 9336
assert set(bayes) == set(xgb)
assert set(maxent).issubset(bayes)

groups = defaultdict(list)
for k, row in maxent.items():
    b = bayes[k]
    assert row["species"] == b["species"]
    scores = (float(b["mean_probability"]), float(row["relative_suitability"]),
              float(xgb[k]["predicted_probability"]))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
    groups[(row["species"], int(row["fold"]))].append((int(row["observed"]), *scores))

def auc(y, p):
    positives = sum(y)
    negatives = len(y) - positives
    assert positives > 0 and negatives > 0
    ranked = sorted(zip(p, y))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(v for _, v in ranked[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def average_precision(y, p):
    order = sorted(range(len(y)), key=lambda i: (-p[i], i))
    found = 0
    total = 0.0
    for rank, idx in enumerate(order, 1):
        if y[idx]:
            found += 1
            total += found / rank
    return total / sum(y)

assert len(matched) == len(groups) == 23
seen = set()
for row in matched:
    cell = (row["species"], int(row["fold"]))
    assert cell not in seen
    seen.add(cell)
    records = groups[cell]
    observed = [v[0] for v in records]
    assert len(records) == int(row["n"])
    assert sum(observed) == int(row["positives"])
    for index, model in enumerate(("bayesian", "maxent", "xgboost"), 1):
        scores = [v[index] for v in records]
        for metric, compute in (("auc", auc), ("ap", average_precision)):
            actual = compute(observed, scores)
            expected = float(row[f"{model}_{metric}"])
            assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (
                cell, model, metric, actual, expected)
    print(f"PASS {cell[0]} fold {cell[1]}: {len(records)} records")
assert seen == set(groups)
print("PASS: all 138 matched AUC/AP metric values reproduced from OOF records")

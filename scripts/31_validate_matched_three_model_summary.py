#!/usr/bin/env python3
"""Independently audit committed matched three-model species means from fold CSV."""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results/matched_bayesian_maxent_xgboost_folds.csv"
SPECIES = ROOT / "results/matched_bayesian_maxent_xgboost_species.csv"
METRICS = ("bayesian_auc", "maxent_auc", "xgboost_auc",
           "bayesian_ap", "maxent_ap", "xgboost_ap")


def load(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    folds, summary = load(FOLDS), load(SPECIES)
    assert len(folds) == 23, f"Expected 23 fold rows, got {len(folds)}"
    assert len(summary) == 5, f"Expected 5 species rows, got {len(summary)}"
    by_species = defaultdict(list)
    seen = set()
    for row in folds:
        key = (row["species"], int(row["fold"]))
        assert key not in seen, f"Duplicate species-fold: {key}"
        seen.add(key)
        n, positives = int(row["n"]), int(row["positives"])
        assert 0 < positives < n
        for metric in METRICS:
            value = float(row[metric])
            assert math.isfinite(value) and 0 <= value <= 1, (key, metric, value)
        by_species[row["species"]].append(row)
    assert len(by_species) == 5
    assert set(by_species) == {row["species"] for row in summary}
    for row in summary:
        sp = row["species"]
        cells = by_species[sp]
        assert int(row["folds"]) == len(cells)
        assert int(row["n"]) == sum(int(c["n"]) for c in cells)
        assert int(row["positives"]) == sum(int(c["positives"]) for c in cells)
        for metric in METRICS:
            expected = sum(float(c[metric]) for c in cells) / len(cells)
            actual = float(row[metric])
            assert math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12), (
                sp, metric, expected, actual)
        print(f"PASS {sp}: {len(cells)} folds; all 6 means and counts match")
    print("PASS: 23 matched species-fold rows; 5 species summaries; no mismatches")


if __name__ == "__main__":
    main()

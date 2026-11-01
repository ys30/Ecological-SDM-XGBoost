#!/usr/bin/env python3
"""Reproduce species-by-spatial-fold presence audit from committed source CSVs."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train.csv"
ASSIGNMENTS = ROOT / "results/spatial_2d_block_assignments.csv"
OUT = ROOT / "results/species_spatial_fold_presence_counts.csv"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    assignments = read_csv(ASSIGNMENTS)
    folds = {}
    for r in assignments:
        plot = r["plot"]
        assert plot not in folds, f"duplicate plot assignment: {plot}"
        fold = int(r["fold"])
        assert 1 <= fold <= 5
        folds[plot] = fold
    assert len(folds) == 1167, f"unexpected plot count: {len(folds)}"

    data = read_csv(SOURCE)
    assert len(data) == 9336, f"unexpected row count: {len(data)}"
    counts = Counter()
    plot_species = set()
    for r in data:
        plot, species = r["plot"], r["Species"]
        assert plot in folds, f"plot missing fold: {plot}"
        key = (plot, species)
        assert key not in plot_species, f"duplicate plot/species: {key}"
        plot_species.add(key)
        outcome = int(r["pres.abs"])
        assert outcome in (0, 1), f"nonbinary outcome: {outcome}"
        counts[(species, folds[plot], "n")] += 1
        counts[(species, folds[plot], "positive")] += outcome

    species = sorted({s for s, _, _ in counts})
    assert len(species) == 8
    rows = []
    for sp in species:
        for fold in range(1, 6):
            n = counts[(sp, fold, "n")]
            positive = counts[(sp, fold, "positive")]
            assert n == sum(f == fold for f in folds.values())
            assert 0 <= positive <= n
            rows.append({"species": sp, "fold": fold, "n": n,
                         "positives": positive, "negatives": n - positive,
                         "auc_evaluable": int(0 < positive < n)})
    assert sum(r["n"] for r in rows) == 9336
    assert sum(r["positives"] for r in rows) == 747

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for sp in species:
        sub = [r for r in rows if r["species"] == sp]
        print(f"{sp}: positives={[r['positives'] for r in sub]}, "
              f"AUC-eligible={sum(r['auc_evaluable'] for r in sub)}/5")
    print(f"PASS: {len(rows)} species-fold cells; {len(data)} records; "
          f"{sum(r['positives'] for r in rows)} presences; {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()

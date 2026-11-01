#!/usr/bin/env python3
"""Compare per-fold rare-species top-k capture with random selection expectation."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/bayesian_rare_species_within_fold_retrieval.csv"
OUTPUT = ROOT / "results/bayesian_rare_species_random_baseline.csv"
KS = (10, 25, 50)

def main():
    by_species = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_species[row["species"]].append(row)
    result = []
    for species in sorted(by_species):
        rows = by_species[species]
        assert len(rows) == 5
        total_positives = sum(int(r["positives"]) for r in rows)
        for k in KS:
            actual = sum(int(r[f"top_{k}_captured"]) for r in rows)
            expected = sum(k * int(r["positives"]) / int(r["n"]) for r in rows)
            result.append({
                "species": species, "top_k_per_fold": k,
                "plots_screened": 5 * k,
                "total_positives": total_positives,
                "actual_captured": actual,
                "expected_random_captured": f"{expected:.8f}",
                "enrichment_vs_random": f"{actual / expected:.8f}" if expected else "",
                "recall": f"{actual / total_positives:.8f}" if total_positives else ""
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(result)
    for row in result:
        print(row)
    print(f"PASS {len(result)} rare-species top-k random-baseline comparisons")

if __name__ == "__main__":
    main()

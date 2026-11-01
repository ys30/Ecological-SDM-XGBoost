#!/usr/bin/env python3
"""Reproduce rare-species Bayesian baseline, pooled and within-fold top-k audits."""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/bayes_spatial_cv_predictions.csv"
OUTPUT = ROOT / "results/bayesian_rare_species_within_fold_retrieval.csv"
RARE = ("Corymbia.gummifera", "Corymbia.intermedia", "Eucalyptus.carnea")
K = (10, 25, 50)

def main():
    grouped = defaultdict(list)
    with INPUT.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["species"] not in RARE:
                continue
            y = int(row["observed"])
            p = float(row["mean_probability"])
            assert y in (0, 1) and math.isfinite(p) and 0 <= p <= 1
            grouped[(row["species"], int(row["fold"]))].append(
                (row["id"], y, p))
    output = []
    for species in RARE:
        for fold in range(1, 6):
            records = grouped[(species, fold)]
            assert len(records) in (229, 233, 234, 237)
            assert len({id for id, _, _ in records}) == len(records)
            ranked = sorted(records, key=lambda r: (-r[2], int(r[0])))
            positives = sum(y for _, y, _ in ranked)
            item = {"species": species, "fold": fold, "n": len(ranked),
                    "positives": positives}
            for k in K:
                captured = sum(y for _, y, _ in ranked[:k])
                item[f"top_{k}_captured"] = captured
                item[f"top_{k}_recall"] = (
                    f"{captured / positives:.8f}" if positives else "")
            output.append(item)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(f"{row['species']} fold {row['fold']}: "
              f"positives={row['positives']} top25={row['top_25_captured']}")
    print(f"PASS {len(output)} species-fold rows; output={OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()

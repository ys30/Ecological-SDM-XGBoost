#!/usr/bin/env python3
"""Inventory out-of-fold prediction CSVs without assuming they exist."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = {"id", "plot", "species", "fold"}
PREDICTION_FIELDS = {"mean_probability", "probability", "predicted_probability", "prediction", "pred", "prob", "p"}
RARE = {"nsw19", "nsw20", "nsw22"}

def main():
    candidates = sorted(RESULTS.rglob("*pred*csv"))
    if not candidates:
        print("No prediction-named CSVs in tracked results; no rare-species score is claimed.")
        return
    usable = 0
    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
                if not REQUIRED.issubset(cols) or not (cols & PREDICTION_FIELDS):
                    print(f"NOT_MATCHED_SCHEMA {path.relative_to(ROOT)}")
                    continue
                counts = {s: 0 for s in RARE}
                keys = set()
                for row in reader:
                    key = (row["id"], row["fold"])
                    if key in keys:
                        raise ValueError(f"Duplicate id/fold: {key}")
                    keys.add(key)
                    if row["species"] in counts:
                        counts[row["species"]] += 1
                usable += 1
                print(f"MATCHED_SCHEMA {path.relative_to(ROOT)} rows={len(keys)} rare_species_rows={counts}")
        except (UnicodeError, csv.Error, OSError, ValueError) as e:
            print(f"INVALID {path.relative_to(ROOT)}: {e}")
            raise
    print(f"Prediction CSV candidates={len(candidates)} matched_schema={usable}")
    print("NOTE: Schema availability alone does not prove all algorithms have matched out-of-fold predictions.")

if __name__ == "__main__":
    main()

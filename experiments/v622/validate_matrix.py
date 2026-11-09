#!/usr/bin/env python3
"""Independent v6.22 model/contrast matrix gate. Does not imply plan freeze."""
from pathlib import Path
import csv, sys
root = Path("experiments/v622/config")
def read(name, required):
    path = root / name
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            sys.exit(f"FAIL {name}: missing columns")
        rows = list(reader)
    if any(None in row or any(v is None or not v.strip() for v in row.values() if v is not None) for row in rows):
        sys.exit(f"FAIL {name}: malformed or empty fields")
    return rows
models = read("v622_model_matrix.csv", ["model_id","family","algorithm","sharing","response","predictors","tuning","schemes","status","notes"])
contrasts = read("v622_contrasts.csv", ["contrast_id","tier","question","model_a","model_b"])
ids = [m["model_id"] for m in models]
cids = [c["contrast_id"] for c in contrasts]
assert len(ids) == len(set(ids)) == 19, "19 unique models required"
assert len(cids) == len(set(cids)) == 20, "20 unique contrasts required"
assert "glmnet_quad_rt" in ids and "C2f" in cids
assert all(c["model_a"] in ids and c["model_b"] in ids and c["model_a"] != c["model_b"] for c in contrasts)
print("PASS: independent model matrix gate (19 models, 20 contrasts). NOT FROZEN.")

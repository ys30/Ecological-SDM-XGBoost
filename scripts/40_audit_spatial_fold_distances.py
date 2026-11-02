#!/usr/bin/env python3
"""Quantify geographic train/test separation of the archived spatial folds.

Distances are great-circle (haversine) km, not projected CRS distances.
No outcome labels or species identities are read.
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "results/spatial_2d_block_assignments.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 1167, len(rows)
assert len({r["plot"] for r in rows}) == len(rows)
assert {int(r["fold"]) for r in rows} == set(range(1, 6))
coords = [(r["plot"], int(r["fold"]), math.radians(float(r["lat"])), math.radians(float(r["long"]))) for r in rows]
assert all(math.isfinite(v) for _, _, lat, lon in coords for v in (lat, lon))
assert all(-math.pi/2 <= lat <= math.pi/2 and -math.pi <= lon <= math.pi for _, _, lat, lon in coords)

def distance(a, b):
    dlat = a[2] - b[2]
    dlon = a[3] - b[3]
    h = math.sin(dlat/2)**2 + math.cos(a[2])*math.cos(b[2])*math.sin(dlon/2)**2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(max(0.0, h))))

output = []
for k in range(1, 6):
    test = [r for r in coords if r[1] == k]
    train = [r for r in coords if r[1] != k]
    assert test and train
    nearest = sorted(min(distance(a, b) for b in train) for a in test)
    n = len(nearest)
    def quantile(p):
        h = (n-1)*p
        i = int(h)
        return nearest[i] + (nearest[min(i+1,n-1)]-nearest[i])*(h-i)
    output.append({"fold":k, "test_plots":n, "train_plots":len(train),
                   "min_nearest_train_km":nearest[0],
                   "p05_nearest_train_km":quantile(.05),
                   "median_nearest_train_km":quantile(.5),
                   "p95_nearest_train_km":quantile(.95),
                   "max_nearest_train_km":nearest[-1],
                   "test_plots_within_1km":sum(v < 1 for v in nearest),
                   "test_plots_within_5km":sum(v < 5 for v in nearest)})
path = ROOT / "results/spatial_fold_nearest_train_distances.csv"
with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(output[0]))
    w.writeheader()
    w.writerows(output)
for row in output:
    print(row)
print("Great-circle plot-centroid distances; adjacent blocks are permitted; not buffered CV.")

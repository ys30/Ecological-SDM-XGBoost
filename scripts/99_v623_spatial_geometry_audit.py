#!/usr/bin/env python3
"""v6.23 geometry-first audit of the frozen v6.22 inner partitions; no fitting.

Construct geographic partitions without labels; join outcomes only afterward.
Reports held-out geographic coverage, closest training distance, and rare-class
feasibility. This diagnoses, but does NOT fix, the infeasible CV design.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

SEED = 622
GROUPS = 4
OUT = Path("results/v623_geometry_audit")
FIELDS = ("outer_fold", "inner_fold", "taxon", "validation_n",
          "validation_positive", "train_n", "train_positive",
          "min_train_distance_km", "median_nearest_train_km",
          "validation_lon_min", "validation_lon_max",
          "validation_lat_min", "validation_lat_max", "status")


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def haversine_km(a, b):
    """Vectorized great-circle distance for lon/lat degrees."""
    lon1, lat1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lon2, lat2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    dlon, dlat = lon2 - lon1, lat2 - lat1
    q = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def main():
    rows = read("data/train.csv")
    fold_rows = read("results/spatial_2d_block_assignments.csv")
    fmap = {r["plot"]: int(r["fold"]) for r in fold_rows}
    if len(fmap) != len(fold_rows):
        raise ValueError("Duplicate outer fold assignments")
    taxa = sorted({r["Species"] for r in rows})
    coords = {}
    labels = {}
    for r in rows:
        p = r["plot"]
        xy = (float(r["long"]), float(r["lat"]))
        if p in coords and coords[p] != xy:
            raise ValueError("Inconsistent plot coordinates: " + p)
        coords[p] = xy
        key = (p, r["Species"])
        if key in labels:
            raise ValueError("Duplicate plot/taxon: " + str(key))
        labels[key] = int(r["pres.abs"])
    if set(coords) != set(fmap) or len(taxa) != 8:
        raise ValueError("Unexpected canonical panel or missing folds")
    if any(not np.isfinite(xy).all() for xy in coords.values()):
        raise ValueError("Nonfinite coordinates")
    records = []
    assignments = []
    for outer in range(1, 6):
        plots = sorted((p for p in fmap if fmap[p] != outer), key=int)
        xy = np.asarray([coords[p] for p in plots])
        groups = KMeans(n_clusters=GROUPS, random_state=SEED, n_init=10).fit_predict(xy)
        for p, g in zip(plots, groups):
            assignments.append({"outer_fold": outer, "plot": p, "inner_fold": int(g)})
        for g in range(GROUPS):
            mask = groups == g
            val = xy[mask]
            train = xy[~mask]
            if not len(val) or not len(train):
                raise ValueError("Empty spatial partition")
            # Geography calculated before outcome labels are consulted.
            distances = haversine_km(val, train).min(axis=1)
            geo = dict(min_train_distance_km=float(distances.min()),
                       median_nearest_train_km=float(np.median(distances)),
                       validation_lon_min=float(val[:, 0].min()),
                       validation_lon_max=float(val[:, 0].max()),
                       validation_lat_min=float(val[:, 1].min()),
                       validation_lat_max=float(val[:, 1].max()))
            for taxon in taxa:
                vp = sum(labels[(p, taxon)] for p, m in zip(plots, mask) if m)
                tp = sum(labels[(p, taxon)] for p, m in zip(plots, mask) if not m)
                vn = int(mask.sum())
                tn = len(plots) - vn
                records.append(dict(outer_fold=outer, inner_fold=g, taxon=taxon,
                                    validation_n=vn, validation_positive=vp,
                                    train_n=tn, train_positive=tp, **geo,
                                    status="PASS" if 0 < tp < tn else "FAIL_SINGLE_CLASS"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("geometry_and_class_audit.csv", records),
                       ("geographic_partition_assignments.csv", assignments)):
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    bad = [r for r in records if r["status"] != "PASS"]
    summary = {"status": "DIAGNOSTIC_ONLY_NOT_A_VALIDATED_PROTOCOL",
               "method": "frozen 4-group KMeans lon/lat; geometry computed before labels",
               "outer_folds": 5, "taxa": len(taxa), "inner_groups": GROUPS,
               "tested_cells": len(records), "failed_cells": len(bad),
               "failures": bad, "min_distance_km_all_cells":
               min(r["min_train_distance_km"] for r in records),
               "warning": "Nearest-neighbor distance is descriptive, NOT a 10-km buffer or leakage proof. No model was trained."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

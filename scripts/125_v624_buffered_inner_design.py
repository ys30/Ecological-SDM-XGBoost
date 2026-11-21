#!/usr/bin/env python3
"""Audit the v6.24 spatial tuning design without fitting or scoring models."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# The distance definition matches the fixed baseline. Keep this diagnostic
# independent of the XGBoost and scikit-learn runtime.
def distances(a, b):
    lon_a, lat_a = np.radians(a).T[:, :, None]
    lon_b, lat_b = np.radians(b).T[:, None, :]
    q = np.sin((lat_b - lat_a) / 2) ** 2
    q += np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def design(rows, fold_rows):
    folds = {row["plot"]: int(row["fold"]) for row in fold_rows}
    coordinates = {}
    outcomes = defaultdict(dict)
    for row in rows:
        plot = row["plot"]
        xy = (float(row["long"]), float(row["lat"]))
        if coordinates.setdefault(plot, xy) != xy or plot in outcomes[row["Species"]]:
            raise ValueError(f"Inconsistent or duplicate record for {plot}")
        outcomes[row["Species"]][plot] = int(row["pres.abs"])
    plots = sorted(coordinates, key=int)
    if set(plots) != set(folds) or set(folds.values()) != set(range(1, 6)):
        raise ValueError("Unexpected plot coverage or outer folds")
    if len(outcomes) != 8 or any(set(labels) != set(plots) for labels in outcomes.values()):
        raise ValueError("Expected a complete eight-taxon panel")

    xy = np.array([coordinates[p] for p in plots])
    fold = np.array([folds[p] for p in plots])
    separation = distances(xy, xy)
    for outer in range(1, 6):
        test = np.flatnonzero(fold == outer)
        pool = np.flatnonzero((fold != outer) & (separation[:, test].min(axis=1) > 10))
        cuts = np.quantile(xy[pool], [.25, .5, .75], axis=0)
        cells = 4 * (xy[pool, 0, None] > cuts[:, 0]).sum(axis=1)
        cells += (xy[pool, 1, None] > cuts[:, 1]).sum(axis=1)
        for taxon, labels in sorted(outcomes.items()):
            y = np.array([labels[p] if folds[p] != outer else -1 for p in plots])
            n_positive = int(y[pool].sum())
            if not 0 < n_positive < len(pool):
                raise ValueError(f"Single-class outer training: {outer}, {taxon}")
            for cell in sorted(set(cells)):
                validation = pool[cells == cell]
                train = pool[separation[np.ix_(pool, validation)].min(axis=1) > 10]
                train_pos, val_pos = int(y[train].sum()), int(y[validation].sum())
                yield dict(outer=outer, taxon=taxon, cell=int(cell),
                           outer_train_positive=n_positive, train_n=len(train),
                           train_positive=train_pos, validation_n=len(validation),
                           validation_positive=val_pos,
                           usable=0 < train_pos < len(train)), train, validation, pool, test


def main():
    import csv
    import hashlib

    source = Path("data/train.csv")
    folds = Path("results/spatial_2d_block_assignments.csv")

    def read(path):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    report = [record for record, *_ in design(read(source), read(folds))]
    grouped = defaultdict(list)
    for record in report:
        grouped[record["outer"], record["taxon"]].append(record)
    eligibility = []
    for (outer, taxon), cells in grouped.items():
        usable = [r for r in cells if r["usable"]]
        positive_cells = sum(r["validation_positive"] > 0 for r in usable)
        eligibility.append(dict(outer=outer, taxon=taxon, usable_cells=len(usable),
                                total_cells=len(cells), positive_cells=positive_cells,
                                outer_train_positive=cells[0]["outer_train_positive"],
                                eligible=len(usable) >= 2 and positive_cells >= 1
                                and cells[0]["outer_train_positive"] >= 5))
    output = Path("results/v624_buffered_inner_design")
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [("cells", report), ("eligibility", eligibility)]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = dict(status="NO_FIT", audited_cells=len(report),
                   invalid_training_cells=sum(not r["usable"] for r in report),
                   ineligible=[r for r in eligibility if not r["eligible"]],
                   data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   fold_sha256=hashlib.sha256(folds.read_bytes()).hexdigest())
    serialized = json.dumps(summary, indent=2) + "\n"
    (output / "summary.json").write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preflight audit for EMS v6.22. No model fitting or outcome evaluation."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--folds", required=True, help="CSV with plot and fold columns; must be the approved primary design")
    p.add_argument("--design", required=True)
    p.add_argument("--output", default="results/v622/preflight.json")
    args = p.parse_args()
    data, folds = load(args.data), load(args.folds)
    required = {"id", "plot", "Species", "pres.abs", "rainann", "soildepth", "tempann", "topo"}
    assert data and required <= set(data[0]), "Missing required input columns"
    assert folds and {"plot", "fold"} <= set(folds[0]), "Missing plot/fold columns"
    ids = [r["id"] for r in data]
    assert len(ids) == len(set(ids)), "Duplicate record IDs"
    assert all(r["pres.abs"] in ("0", "1") for r in data), "Nonbinary outcome"
    assert all(all(r[k] not in ("", "NA", "NaN") for k in ("rainann", "soildepth", "tempann", "topo")) for r in data), "Missing predictor"
    mapping = {}
    for r in folds:
        if r["plot"] in mapping:
            assert mapping[r["plot"]] == r["fold"], "Conflicting plot assignments"
        mapping[r["plot"]] = r["fold"]
    plots = {r["plot"] for r in data}
    assert plots == set(mapping), "Plot coverage mismatch: missing or extraneous plots"
    taxa = sorted({r["Species"] for r in data})
    groups = defaultdict(lambda: [0, 0])
    for r in data:
        k = (r["Species"], mapping[r["plot"]])
        groups[k][0] += 1
        groups[k][1] += int(r["pres.abs"])
    counts = Counter(mapping.values())
    report = {
        "status": "PREFLIGHT_ONLY_NOT_EXPERIMENTAL_RESULTS",
        "design": args.design,
        "input_sha256": sha(args.data),
        "fold_sha256": sha(args.folds),
        "records": len(data), "plots": len(plots), "taxa": taxa,
        "detections": sum(int(r["pres.abs"]) for r in data),
        "fold_plot_counts": dict(sorted(counts.items())),
        "taxon_fold_counts": [
            {"taxon": t, "fold": f, "n": v[0], "detections": v[1],
             "auc_evaluable": 0 < v[1] < v[0]}
            for (t, f), v in sorted(groups.items())
        ],
        "warnings": [
            "This script does not verify CRS, physical buffer distances or how folds were generated.",
            "Confirm the supplied fold file corresponds to the prespecified design before freezing.",
            "Inner folds must be generated exclusively within each outer training set."
        ]
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(data)} rows, {len(plots)} plots, {len(taxa)} taxa, {len(counts)} folds")
    print(f"Preflight report: {out}")

if __name__ == "__main__":
    main()

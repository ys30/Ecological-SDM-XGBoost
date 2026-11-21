"""Pair comparators by observation key and verify their buffered memberships."""
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

fixed = importlib.import_module("102_v623_fixed_xgb_outer_oof")
analysis = importlib.import_module("127_v624_c1_analysis")
base = Path("results/v624_comparators")
out = Path("results/v624_matched_analysis")
out.mkdir(parents=True, exist_ok=True)
canonical = {r["id"]: r for r in fixed.read("data/train.csv")}
membership = defaultdict(set)
for r in fixed.read(base / "membership.csv"):
    membership[r["outer"], r["role"]].add(r["plot"])
train_hash = {}
for fold in range(1, 6):
    for taxon in sorted({r["Species"] for r in canonical.values()}):
        train_hash[str(fold), taxon] = fixed.ids_sha256(
            k for k, r in canonical.items()
            if r["plot"] in membership[str(fold), "train"] and r["Species"] == taxon
        )
arms = defaultdict(dict)


def insert(arm, row, probability):
    rid = row["id"]
    identity = row["plot"], row["taxon"], row["y"]
    original = canonical[rid]
    if identity != (original["plot"], original["Species"], original["pres.abs"]):
        raise ValueError(f"Source row mismatch for {rid}")
    if row["plot"] not in membership[row["outer"], "test"]:
        raise ValueError(f"Wrong outer fold for {rid}")
    if rid in arms[arm] or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Duplicate or invalid prediction: {arm}/{rid}")
    arms[arm][rid] = row["outer"], row["taxon"], int(row["y"]), probability


for r in fixed.read(base / "predictions.csv"):
    insert(r["arm"], r, float(r["p"]))
for fold in range(1, 6):
    for r in fixed.read(Path(f"results/v624_c1/622/fold{fold}/paired_oof.csv")):
        if r["train_sha256"] != train_hash[r["outer"], r["taxon"]]:
            raise ValueError("Training rows are not matched")
        insert("xgb_tuned_policy", r, float(r["tuned"]))
        insert("xgb_fixed", r, float(r["fixed"]))

comparisons = [
    ("C2", f"{level}_quadratic", f"{level}_linear")
    for level in ("no_sharing", "partial_slopes", "full_slopes")
] + [
    ("C3", "partial_slopes_quadratic", "no_sharing_quadratic"),
    ("C3", "full_slopes_quadratic", "no_sharing_quadratic"),
    ("C3", "full_slopes_quadratic", "partial_slopes_quadratic"),
    ("C4", "xgb_tuned_policy", "no_sharing_quadratic"),
    ("C4", "xgb_tuned_policy", "maxent_fixed"),
    ("C4", "maxent_fixed", "no_sharing_quadratic"),
]
metrics = []
for contrast, left, right in comparisons:
    shared = sorted(arms[left].keys() & arms[right].keys())
    if "maxent_fixed" not in (left, right) and len(shared) != len(canonical):
        raise ValueError(f"Incomplete matched contrast: {left}/{right}")
    cells = defaultdict(list)
    for rid in shared:
        fold, taxon, y, score_left = arms[left][rid]
        fold_r, taxon_r, y_r, score_right = arms[right][rid]
        if (fold, taxon, y) != (fold_r, taxon_r, y_r):
            raise ValueError("Unmatched evaluation row")
        cells[fold, taxon].append((y, score_left, score_right))
    for (fold, taxon), observations in sorted(cells.items()):
        y, p, q = np.asarray(observations).T
        eligible = len(np.unique(y)) == 2
        probabilistic = "maxent_fixed" not in (left, right)
        metrics.append(dict(
            contrast=contrast, left=left, right=right, fold=fold, taxon=taxon,
            n=len(y), positives=int(y.sum()),
            auc_difference=roc_auc_score(y, p) - roc_auc_score(y, q) if eligible else "",
            ap_difference=average_precision_score(y, p) - average_precision_score(y, q) if eligible else "",
            logloss_difference=(analysis.point_loss(y, p) - analysis.point_loss(y, q)).mean()
            if probabilistic else "",
            brier_difference=((y - p) ** 2 - (y - q) ** 2).mean() if probabilistic else "",
        ))
fixed.write(out / "matched_taxon_fold_differences.csv", metrics)
summary = []
fields = ("auc_difference", "ap_difference", "logloss_difference", "brier_difference")
for contrast, left, right in comparisons:
    matching = [r for r in metrics if (r["contrast"], r["left"], r["right"]) == (contrast, left, right)]
    for field in fields:
        valid = [r[field] for r in matching if r[field] != ""]
        if valid:
            summary.append(dict(contrast=contrast, left=left, right=right, metric=field,
                                cells=len(valid), mean=float(np.mean(valid))))
fixed.write(out / "matched_summary.csv", summary)
print(json.dumps({arm: len(values) for arm, values in arms.items()}, indent=2))
print((out / "matched_summary.csv").read_text())

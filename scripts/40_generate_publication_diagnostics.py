#!/usr/bin/env python3
"""Create publication candidate maps and matched-fold diagnostics from committed CSVs.
Only anonymized species identifiers are displayed.
"""
import csv
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
def read(path):
    with (root / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
sites = read("results/spatial_2d_block_assignments.csv")
folds = read("results/matched_bayesian_maxent_xgboost_folds.csv")
assert len(sites) == 1167 and len(folds) == 23
assert len({int(r["plot"]) for r in sites}) == len(sites)
assert set(int(r["fold"]) for r in sites) == set(range(1, 6))
colors = ["#2166ac", "#67a9cf", "#1b7837", "#f4a582", "#b2182b"]
fig, ax = plt.subplots(figsize=(8, 7))
counts = Counter(int(r["fold"]) for r in sites)
for fold in range(1, 6):
    group = [r for r in sites if int(r["fold"]) == fold]
    ax.scatter([float(r["long"]) for r in group], [float(r["lat"]) for r in group],
               s=13, alpha=0.72, color=colors[fold-1], label=f"Fold {fold} (n={counts[fold]})", linewidths=0)
ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="NSW survey plots by spatial holdout fold")
ax.legend(frameon=False, loc="best", fontsize=9)
ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig(out / "40_spatial_holdout_map.svg")
plt.close(fig)

# Fold-level paired differences preserve shared held-out observations;
# no independent-fold confidence interval is inferred.
models = [("Bayesian", "bayesian"), ("MaxEnt", "maxent"), ("XGBoost", "xgboost")]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, metric in zip(axes, ["auc", "ap"]):
    labels = sorted({r["species"] for r in folds})
    for i, species in enumerate(labels):
        sub = [r for r in folds if r["species"] == species]
        for j, (name, key) in enumerate(models):
            xs = [float(r[f"{key}_{metric}"]) for r in sub]
            ys = [i + (j-1)*.22] * len(xs)
            ax.scatter(xs, ys, s=23, alpha=.65, color=colors[j], marker="o")
            ax.scatter(sum(xs)/len(xs), i+(j-1)*.22, s=75, color=colors[j], marker="|", linewidths=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC" if metric == "auc" else "Average precision")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(axis="x", alpha=.2)
axes[0].invert_yaxis()
fig.suptitle("Matched held-out fold performance: points = folds; ticks = fold means")
fig.text(.5, .015, "Descriptive variation; fold observations are not independent replications", ha="center", fontsize=9)
fig.tight_layout(rect=[0,.035,1,.94])
fig.savefig(out / "41_matched_fold_performance.svg")
plt.close(fig)
print("PASS: created 40_spatial_holdout_map.svg and 41_matched_fold_performance.svg")

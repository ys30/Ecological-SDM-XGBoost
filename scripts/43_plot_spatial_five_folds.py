#!/usr/bin/env python3
"""Plot recorded spatial-fold assignments, without refitting SDMs or changing assignments."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
source=root/"results/spatial_2d_block_assignments.csv"
with source.open(newline="") as f:
    rows=list(csv.DictReader(f))
assert len(rows)==1167, f"Unexpected plot count: {len(rows)}"
assert len({r["plot"] for r in rows})==len(rows)
assert sorted({int(r["fold"]) for r in rows})==[1,2,3,4,5]
fig,ax=plt.subplots(figsize=(8.2,6.3))
for fold in range(1,6):
    subset=[r for r in rows if int(r["fold"])==fold]
    ax.scatter([float(r["long"]) for r in subset],[float(r["lat"]) for r in subset],s=10,alpha=.75,label=f"Fold {fold} (n={len(subset)})",rasterized=True)
ax.set(xlabel="Longitude (degrees east)",ylabel="Latitude (degrees north)",title="Observed sampling plots and five spatial holdout folds")
ax.legend(loc="best",fontsize=8,frameon=True)
ax.set_aspect("equal",adjustable="datalim")
ax.grid(alpha=.16)
fig.tight_layout()
out=root/"results/figures/25_spatial_five_fold_map.png"
fig.savefig(out,dpi=220)
print(f"Saved {out}; {len(rows)} distinct plots")

"""Export outer training memberships for matched R comparators."""
import importlib
from pathlib import Path

fixed = importlib.import_module("102_v623_fixed_xgb_outer_oof")
audit = importlib.import_module("125_v624_buffered_inner_design")
rows = fixed.read("data/train.csv")
folds = fixed.read("results/spatial_2d_block_assignments.csv")
plots = sorted({r["plot"] for r in rows}, key=int)
seen = set()
membership = []
for record, _, _, train, test in audit.design(rows, folds):
    fold = record["outer"]
    if fold in seen:
        continue
    seen.add(fold)
    for role, indices in (("train", train), ("test", test)):
        membership.extend(dict(outer=fold, plot=plots[i], role=role) for i in indices)
output = Path("results/v624_comparators")
output.mkdir(parents=True, exist_ok=True)
fixed.write(output / "membership.csv", membership)

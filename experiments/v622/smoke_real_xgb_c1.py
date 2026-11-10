"""Real-input C1 runtime smoke: one outer grid fold, one XGB candidate.

Not a preregistered model-selection run: uses a single candidate and never
reads outer-test labels for fitting or scoring. No manuscript metrics emitted.
"""
import hashlib
from pathlib import Path
import time
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.xgb_inner_runner import run_c1_inner_tuning
from experiments.v622.xgb_backend import XGBoostC1Backend
from experiments.v622.xgb_candidate_sampler import generate_candidates

INPUTS = {
    "data/train.csv": "f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
    "results/spatial_2d_block_assignments.csv": "4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
}
for path, expected in INPUTS.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"SHA mismatch: {path}"

data = pd.read_csv("data/train.csv")
assignments = pd.read_csv("results/spatial_2d_block_assignments.csv")
assert assignments["plot"].is_unique and assignments["fold"].nunique() == 5
assert set(assignments["plot"]) == set(data["plot"])
outer_fold = sorted(assignments["fold"].unique())[0]
outer_plots = set(assignments.loc[assignments["fold"] == outer_fold, "plot"])
outer_train = data.loc[~data["plot"].isin(outer_plots)].copy()
assert not set(outer_train["plot"]) & outer_plots
inner = spatial_inner_folds(data, outer_plots, outer_fold=int(outer_fold))
inner_map = dict(zip(inner["plot"], inner["inner_fold"]))
assert set(inner_map) == set(outer_train["plot"])
candidate = generate_candidates()[0]
print(f"START real-input C1 smoke outer_fold={outer_fold}, outer_training_plots={len(inner_map)}, candidates=1",flush=True)
for variant in ("species","pooled"):
    backend = XGBoostC1Backend(variant)
    start = time.monotonic()
    result = run_c1_inner_tuning(outer_train,inner_map,backend.primary,backend.retry,
                                 variant=variant,candidates=[candidate])
    statuses = {}
    for cell in result["cell_diagnostics"]:
        statuses[cell["status"]] = statuses.get(cell["status"],0)+1
    print(f"PASS variant={variant} fits={backend.fit_count} cells={len(result['cell_diagnostics'])} "
          f"status={statuses} elapsed_s={time.monotonic()-start:.2f}",flush=True)
print("LIMITATION: single candidate; no outer-test predictions or scientific performance metrics.")

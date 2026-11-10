"""Pre-freeze engineering runtime profile of the full C1 30-candidate search.

This is NOT a formal tuning run: selected candidates, validation losses,
outer-test predictions, and scientific metrics are not exported. Only fitting
counts, statuses, and runtime are printed.
"""
import hashlib
from pathlib import Path
import time
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.xgb_inner_runner import run_c1_inner_tuning
from experiments.v622.xgb_backend import XGBoostC1Backend
from experiments.v622.xgb_candidate_sampler import generate_candidates, candidate_sha256

INPUTS={
 "data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
 "results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
}
for path,sha in INPUTS.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
data=pd.read_csv("data/train.csv")
assign=pd.read_csv("results/spatial_2d_block_assignments.csv")
assert assign["plot"].is_unique and assign["fold"].nunique()==5
assert set(assign["plot"])==set(data["plot"])
fold=sorted(assign["fold"].unique())[0]
outer=set(assign.loc[assign["fold"]==fold,"plot"])
train=data.loc[~data["plot"].isin(outer)].copy()
inner=spatial_inner_folds(data,outer,outer_fold=int(fold))
mapping=dict(zip(inner["plot"],inner["inner_fold"]))
assert set(mapping)==set(train["plot"])
candidates=generate_candidates()
assert len(candidates)==30
print(f"START ENGINEERING PROFILE outer_fold={fold} candidates=30 hash={candidate_sha256()} training_plots={len(mapping)}",flush=True)
for variant in ("species","pooled"):
    backend=XGBoostC1Backend(variant,seed=100+int(fold))
    t=time.monotonic()
    result=run_c1_inner_tuning(train,mapping,backend.primary,backend.retry,
                               variant=variant,candidates=candidates)
    statuses={}
    for row in result["cell_diagnostics"]:
        statuses[row["status"]]=statuses.get(row["status"],0)+1
    assert len(result["candidate_scores"])==30
    assert len(result["cell_diagnostics"])==30*4*train["Species"].nunique()
    assert all(row["status"] in ("fit_success","retry_success","insufficient_positives","fit_failure_fallback") for row in result["cell_diagnostics"])
    print(f"PASS variant={variant} candidates=30 fits={backend.fit_count} "
          f"cells={len(result['cell_diagnostics'])} status={statuses} "
          f"elapsed_s={time.monotonic()-t:.2f}",flush=True)
print("ENGINEERING ONLY: no selected candidate, scores, outer predictions, or scientific metrics exported.",flush=True)

"""Audit real grid spatial folds and train-only nested data preparation.

This is a non-training smoke test; it does not claim that km15 files exist.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.v622.nested_fold_data import prepare_nested_fold,validate_plot_folds

def verify(path,expected):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert actual==expected, f"input hash mismatch {path}: {actual}"

verify("data/train.csv","f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763")
verify("results/spatial_2d_block_assignments.csv","4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81")
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/spatial_2d_block_assignments.csv")
assert a["plot"].is_unique and a["plot"].notna().all()
assert set(a["plot"])==set(d["plot"])
assert a["fold"].nunique()==5
assert a["fold"].notna().all()
assert d.groupby("plot").size().nunique()==1
# This smoke test deliberately uses a deterministic plot-only assignment
# to exercise the data plumbing; it is NOT the preregistered kmeans inner folds.
outer_fold=sorted(a["fold"].unique())[0]
outer=set(a.loc[a["fold"]==outer_fold,"plot"])
remaining=sorted(set(d["plot"])-outer)
inner_map={plot:i%4 for i,plot in enumerate(remaining)}
validate_plot_folds(d,outer,inner_map)
inner_val={plot for plot,f in inner_map.items() if f==0}
p=prepare_nested_fold(d,outer,inner_val)
assert len(p.inner_train_rows)+len(p.inner_validation_rows)+len(p.outer_test_rows)==len(d)
assert p.inner_train_x.index.intersection(p.outer_test_x.index).empty
assert p.inner_validation_x.index.intersection(p.outer_test_x.index).empty
print(f"PASS grid fold smoke: rows={len(d)}, plots={len(a)}, outer_fold={outer_fold}, features={len(p.columns)}")
print("LIMITATION: inner folds are synthetic plot-only plumbing test, not spatial kmeans.")

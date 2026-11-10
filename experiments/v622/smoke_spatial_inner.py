"""Real-data, outer-training-only spatial inner-fold smoke test; no model fitting."""
import hashlib
from pathlib import Path
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.nested_fold_data import prepare_nested_fold,validate_plot_folds

def verify(path,expected):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, f"SHA mismatch: {path}"
verify("data/train.csv","f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763")
verify("results/spatial_2d_block_assignments.csv","4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81")
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/spatial_2d_block_assignments.csv")
outer_fold=sorted(a["fold"].unique())[0]
outer=set(a.loc[a["fold"]==outer_fold,"plot"])
inner=spatial_inner_folds(d,outer,n_clusters=12,n_folds=4,seed=2026)
mapping=dict(zip(inner["plot"],inner["inner_fold"]))
validate_plot_folds(d,outer,mapping)
for f in sorted(inner["inner_fold"].unique()):
    val=set(inner.loc[inner["inner_fold"]==f,"plot"])
    p=prepare_nested_fold(d,outer,val)
    assert len(p.inner_train_rows)+len(p.inner_validation_rows)+len(p.outer_test_rows)==len(d)
    print(f"inner_fold={f} train_rows={len(p.inner_train_rows)} val_rows={len(p.inner_validation_rows)} test_rows={len(p.outer_test_rows)}")
print(f"PASS real-data spatial inner folds: outer={outer_fold}, clusters={inner.cluster.nunique()}, folds={inner.inner_fold.nunique()}")
print("LIMITATION: single outer fold smoke; no tuning, no model scoring, no formal CRS verification.")

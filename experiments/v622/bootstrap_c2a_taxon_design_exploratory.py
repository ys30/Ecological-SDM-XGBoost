"""Descriptive taxon-cluster bootstrap for C2a diagnostic sensitivity.

Resample taxa (not folds) to avoid treating repeated folds as independent taxa.
Bootstrap is exploratory only: preregistration is not frozen.
"""
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
d=pd.read_csv(root/"taxon_design_robustness_DIAGNOSTIC_ONLY.csv")
wide=d.pivot(index="taxon",columns="design",values="mean_delta_auc")
counts=d.pivot(index="taxon",columns="design",values="valid_auc_folds")
rng=np.random.default_rng(622)
rows=[]
for minfold in [1,3,5]:
    eligible=wide[(counts[["grid4","km15"]]>=minfold).all(axis=1)].dropna()
    n=len(eligible)
    if n<3:
        rows.append(dict(min_valid_folds_both_designs=minfold,taxa=n,metric="grid4_minus_km15",
            estimate=np.nan,ci_low=np.nan,ci_high=np.nan,bootstrap_draws=0))
        continue
    a=eligible["grid4"].to_numpy()
    b=eligible["km15"].to_numpy()
    ix=rng.integers(0,n,size=(2000,n))
    for metric,values in [("grid4",a),("km15",b),("grid4_minus_km15",a-b)]:
        boot=values[ix].mean(axis=1)
        rows.append(dict(min_valid_folds_both_designs=minfold,taxa=n,metric=metric,
            estimate=float(values.mean()),ci_low=float(np.quantile(boot,.025)),
            ci_high=float(np.quantile(boot,.975)),bootstrap_draws=2000))
out=pd.DataFrame(rows)
out.to_csv(root/"taxon_cluster_bootstrap_EXPLORATORY_ONLY.csv",index=False)
print("PASS exploratory taxon cluster bootstrap, seed 622, 2000 draws",flush=True)
print(out.to_string(index=False),flush=True)
print("EXPLORATORY ONLY: no confirmatory CI claim; only taxa with both-design coverage",flush=True)

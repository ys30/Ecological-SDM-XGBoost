"""Separate genuine lambda-boundary selections from fallback arms.

A selected progress value is not an interpretable lambda selection when
outer fitting falls back to a constant predictor.
"""
from pathlib import Path
import pandas as pd
root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
d=pd.read_csv(root/"lambda_boundary_and_inner_fallback_DIAGNOSTIC_ONLY.csv")
if len(d)!=160:raise RuntimeError("expected 160 arms")
for c in ["outer_fallback","lower_lambda_boundary","upper_lambda_boundary"]:
    d[c]=d[c].astype(str).str.lower().eq("true")
valid=d[~d.outer_fallback].copy()
rows=[]
for (design,model),g in d.groupby(["design","model"]):
    v=g[~g.outer_fallback]
    rows.append(dict(design=design,model=model,total_arms=len(g),nonfallback_arms=len(v),
        outer_fallback_arms=int(g.outer_fallback.sum()),
        lower_boundary_nonfallback=int(v.lower_lambda_boundary.sum()),
        upper_boundary_nonfallback=int(v.upper_lambda_boundary.sum()),
        lower_boundary_fraction_nonfallback=float(v.lower_lambda_boundary.mean()),
        inner_fallback_affected_nonfallback=int((v.inner_fallback_rows>0).sum()),
        lower_boundary_with_inner_fallback=int((v.lower_lambda_boundary&(v.inner_fallback_rows>0)).sum())))
out=pd.DataFrame(rows)
out.to_csv(root/"lambda_nonfallback_boundary_sensitivity_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS nonfallback-only lambda boundary sensitivity",flush=True)
print(out.to_string(index=False),flush=True)
print("DIAGNOSTIC ONLY: do not interpret selected lambda in constant-prediction fallback arms",flush=True)

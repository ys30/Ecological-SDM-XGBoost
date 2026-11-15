"""Audit selected glmnet lambda-path boundaries from archived C2a diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
paths=sorted(root.rglob("outer_train_only_path_DIAGNOSTIC_ONLY.csv"))
if len(paths)!=80:raise RuntimeError(f"expected 80 paired OOF files, got {len(paths)}")
rows=[]
for path in paths:
    d=pd.read_csv(path)
    design="km15" if "/km15/" in str(path) else "grid4"
    for model,g in d.groupby("model_id"):
        progress=g.lambda_progress.unique()
        fallback=g.fallback.astype(str).str.lower().eq("true").unique()
        if len(progress)!=1 or len(fallback)!=1:raise RuntimeError(f"inconsistent fit metadata {path}")
        trace_path=path.parent/(model+"_train_only_path_trace.csv")
        if not trace_path.exists():raise RuntimeError(f"missing trace {trace_path}")
        tr=pd.read_csv(trace_path)
        if len(tr)!=360 or tr.inner_fold.nunique()!=4:raise RuntimeError(f"bad trace {trace_path}")
        p=float(progress[0])
        rows.append(dict(design=design,fold=int(g.outer_fold.iloc[0]),taxon=str(g.Species.iloc[0]),
            model=model,outer_fallback=bool(fallback[0]),selected_progress=p,
            lower_lambda_boundary=bool(np.isclose(p,1)),upper_lambda_boundary=bool(np.isclose(p,0)),
            inner_fallback_rows=int(tr.fallback.astype(str).str.lower().eq("true").sum()),
            inner_fallback_folds=int(tr.loc[tr.fallback.astype(str).str.lower().eq("true"),"inner_fold"].nunique()),
            selected_alpha=float(g.alpha.iloc[0])))
out=pd.DataFrame(rows).sort_values(["design","fold","taxon","model"])
if len(out)!=160 or out.duplicated(["design","fold","taxon","model"]).any():
    raise RuntimeError("incomplete model metadata")
out.to_csv(root/"lambda_boundary_and_inner_fallback_DIAGNOSTIC_ONLY.csv",index=False)
s=out.groupby(["design","model"]).agg(arms=("taxon","size"),
    outer_fallback=("outer_fallback","sum"),lower_lambda_boundary=("lower_lambda_boundary","sum"),
    upper_lambda_boundary=("upper_lambda_boundary","sum"),
    arms_with_inner_fallback=("inner_fallback_rows",lambda x:int((x>0).sum())),
    mean_selected_progress=("selected_progress","mean")).reset_index()
s.to_csv(root/"lambda_boundary_summary_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS 160-arm lambda path and inner fallback audit",flush=True)
print(s.to_string(index=False),flush=True)
print("DIAGNOSTIC ONLY: boundary selection warrants tuning-path sensitivity; inner fallback rows are not independent folds",flush=True)

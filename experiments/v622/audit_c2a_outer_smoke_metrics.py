"""Audit diagnostic C2a outer smoke predictions; never label as formal inference."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

def main():
    p=Path("results/v622/c2a_outer_smoke/outer_smoke_oof_DIAGNOSTIC_ONLY.csv")
    d=pd.read_csv(p)
    if len(d)!=474 or set(d.model_id)!={"glmnet_lin","glmnet_quad_rt"}:raise RuntimeError("invalid smoke arms")
    if d.duplicated(["plot","Species","outer_fold","model_id"]).any():raise RuntimeError("duplicate predictions")
    if not d.diagnostic_only.astype(str).str.lower().eq("true").all():raise RuntimeError("missing diagnostic marker")
    if not np.isfinite(d.pred).all() or not d.pred.between(0,1).all():raise RuntimeError("invalid probability")
    a=d.loc[d.model_id=="glmnet_lin"].set_index("plot").sort_index()
    b=d.loc[d.model_id=="glmnet_quad_rt"].set_index("plot").sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.y,b.y):raise RuntimeError("unpaired outer observations")
    if len(a)!=237 or a.Species.nunique()!=1 or a.outer_fold.nunique()!=1:raise RuntimeError("wrong held-out coverage")
    rows=[]
    for model,part in [("glmnet_lin",a),("glmnet_quad_rt",b)]:
        y=part.y.to_numpy();pred=np.clip(part.pred.to_numpy(),1e-6,1-1e-6)
        rows.append(dict(model=model,n=len(y),positives=int(y.sum()),auc=float(roc_auc_score(y,pred)) if len(np.unique(y))==2 else np.nan,
                         ap=float(average_precision_score(y,pred)) if len(np.unique(y))==2 else np.nan,
                         log_loss=float(log_loss(y,pred,labels=[0,1])),brier=float(brier_score_loss(y,pred))))
    out=pd.DataFrame(rows)
    out["diagnostic_only"]=True
    target=p.parent/"outer_smoke_metrics_DIAGNOSTIC_ONLY.csv"
    out.to_csv(target,index=False)
    print("PASS paired C2a diagnostic outer metrics; NOT confirmatory",flush=True)
    print(out.to_string(index=False),flush=True)
if __name__=="__main__":main()

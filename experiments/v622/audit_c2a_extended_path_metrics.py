"""Held-out metrics for C2a extended lambda sensitivity (diagnostic only)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
base=Path("results/v622/c2a_outer_smoke")
d=pd.read_csv(base/"outer_extended_path_DIAGNOSTIC_ONLY.csv")
if len(d)!=474 or set(d.model_id)!={"glmnet_lin","glmnet_quad_rt"}:
    raise RuntimeError("incorrect extended-path OOF arms")
if not d.diagnostic_only.astype(str).str.lower().eq("true").all():
    raise RuntimeError("diagnostic marker missing")
if d.duplicated(["plot","model_id"]).any() or not np.isfinite(d.pred).all() or not d.pred.between(0,1).all():
    raise RuntimeError("invalid extended-path predictions")
a=d.loc[d.model_id=="glmnet_lin"].set_index("plot").sort_index()
b=d.loc[d.model_id=="glmnet_quad_rt"].set_index("plot").sort_index()
if len(a)!=237 or not a.index.equals(b.index) or not np.array_equal(a.y,b.y):
    raise RuntimeError("unpaired extended-path OOF")
records=[]
for model,part in [("glmnet_lin",a),("glmnet_quad_rt",b)]:
    y=part.y.to_numpy();p=np.clip(part.pred.to_numpy(),1e-6,1-1e-6)
    progress=part.lambda_progress.to_numpy(float)
    if not np.allclose(progress,progress[0]):raise RuntimeError("inconsistent lambda progress")
    records.append(dict(model=model,n=len(y),positives=int(y.sum()),
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
        ap=float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan,
        log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
        lambda_progress=float(progress[0]),at_lower_boundary=bool(np.isclose(progress[0],1)),
        diagnostic_only=True))
out=pd.DataFrame(records)
out.to_csv(base/"extended_path_metrics_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS C2a extended-path held-out metrics (DIAGNOSTIC ONLY)",flush=True)
print(out.to_string(index=False),flush=True)
if out.at_lower_boundary.any():print("WARNING extended path still selects lower lambda boundary",flush=True)

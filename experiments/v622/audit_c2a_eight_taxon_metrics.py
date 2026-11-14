"""Audit eight-taxon paired C2a OOF and separate fallback arms (diagnostic only)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
root=Path("results/v622/c2a_batch_DIAGNOSTIC_ONLY/grid4_s2026_b0/fold1")
paths=sorted(root.glob("*/outer_train_only_path_DIAGNOSTIC_ONLY.csv"))
if len(paths)!=8:raise RuntimeError(f"expected eight taxa, got {len(paths)}")
records=[]
for path in paths:
    d=pd.read_csv(path)
    if len(d)!=474 or d.duplicated(["plot","model_id"]).any():raise RuntimeError(f"invalid OOF {path}")
    if set(d.model_id)!={"glmnet_lin","glmnet_quad_rt"} or d.plot.nunique()!=237:
        raise RuntimeError(f"unpaired OOF {path}")
    if not d.diagnostic_only.astype(str).str.lower().eq("true").all():
        raise RuntimeError(f"missing diagnostic tag {path}")
    for model,part in d.groupby("model_id"):
        y=part.y.to_numpy();p=part.pred.to_numpy()
        if len(part)!=237 or not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):
            raise RuntimeError(f"bad probabilities {path}")
        if part.fallback.astype(str).str.lower().nunique()!=1:raise RuntimeError("mixed outer fallback")
        fallback=part.fallback.astype(str).str.lower().eq("true").all()
        records.append(dict(taxon=path.parent.name,model=model,n=len(y),positives=int(y.sum()),
            outer_fallback=bool(fallback),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,
            ap=float(average_precision_score(y,p)) if len(set(y))==2 else np.nan,
            log_loss=float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),
            brier=float(brier_score_loss(y,p)),diagnostic_only=True))
    a=d[d.model_id=="glmnet_lin"].set_index("plot").sort_index()
    b=d[d.model_id=="glmnet_quad_rt"].set_index("plot").sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.y,b.y):
        raise RuntimeError(f"unpaired labels {path}")
out=pd.DataFrame(records).sort_values(["taxon","model"])
if len(out)!=16:raise RuntimeError("missing model arms")
out.to_csv(root/"eight_taxon_metrics_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS eight-taxon metrics; model arms",len(out),"outer fallback arms",int(out.outer_fallback.sum()),flush=True)
print(out.to_string(index=False),flush=True)
valid=out[~out.outer_fallback]
print("NON-FALLBACK MODEL ARMS",len(valid),"; fallback excluded from comparative model summaries",flush=True)

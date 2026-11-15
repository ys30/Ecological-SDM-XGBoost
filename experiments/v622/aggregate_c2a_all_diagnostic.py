"""Aggregate diagnostic OOF from all 10 C2a spatial-design/outer-fold cells."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
paths=sorted(root.rglob("outer_train_only_path_DIAGNOSTIC_ONLY.csv"))
if len(paths)!=80:raise RuntimeError(f"expected 80 taxon-fold-design OOF files, got {len(paths)}")
rows=[]
for path in paths:
    d=pd.read_csv(path)
    if len(d)==0 or d.duplicated(["plot","model_id"]).any():raise RuntimeError(f"bad OOF {path}")
    if set(d.model_id)!={"glmnet_lin","glmnet_quad_rt"}:raise RuntimeError(f"missing model {path}")
    if not d.diagnostic_only.astype(str).str.lower().eq("true").all():raise RuntimeError(f"unlabeled OOF {path}")
    taxon=d.Species.unique();fold=d.outer_fold.unique()
    if len(taxon)!=1 or len(fold)!=1 or not d.groupby("plot").model_id.nunique().eq(2).all():
        raise RuntimeError(f"nonpaired OOF {path}")
    design="km15" if "km15" in str(path) else "grid4"
    for model,g in d.groupby("model_id"):
        y=g.y.to_numpy();p=g.pred.to_numpy()
        if not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):raise RuntimeError(f"bad predictions {path}")
        rows.append(dict(design=design,fold=int(fold[0]),taxon=str(taxon[0]),model=model,
            n=len(y),positives=int(y.sum()),fallback=bool(g.fallback.astype(str).str.lower().eq("true").all()),
            auc=roc_auc_score(y,p) if len(set(y))==2 else np.nan,
            ap=average_precision_score(y,p) if len(set(y))==2 else np.nan,
            log_loss=log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]),
            brier=brier_score_loss(y,p),diagnostic_only=True))
out=pd.DataFrame(rows).sort_values(["design","fold","taxon","model"])
if len(out)!=160 or out.duplicated(["design","fold","taxon","model"]).any():raise RuntimeError("missing/duplicate arms")
if set(out.design)!={"grid4","km15"} or set(out.fold)!=set(range(1,6)) or out.taxon.nunique()!=8:
    raise RuntimeError("incomplete experiment")
out.to_csv(root/"all_arm_metrics_DIAGNOSTIC_ONLY.csv",index=False)
key=["design","fold","taxon"]
a=out[out.model=="glmnet_lin"].set_index(key).sort_index()
b=out[out.model=="glmnet_quad_rt"].set_index(key).sort_index()
if not a.index.equals(b.index) or not np.array_equal(a.positives,b.positives):raise RuntimeError("unpaired arm counts")
paired=pd.DataFrame(index=a.index)
paired["positives"]=a.positives
paired["n"]=a.n
paired["any_fallback"]=a.fallback|b.fallback
paired["auc_comparable"]=(~paired.any_fallback)&paired.positives.gt(0)&paired.positives.lt(paired.n)
for metric in ["auc","ap","log_loss","brier"]:
    paired["delta_"+metric]=b[metric]-a[metric]
paired=paired.reset_index()
paired.to_csv(root/"paired_cell_metrics_DIAGNOSTIC_ONLY.csv",index=False)
valid=paired[paired.auc_comparable]
print("PASS 160 arms / 80 paired cells, comparable AUC cells",len(valid),
      "fallback cells",int(paired.any_fallback.sum()),
      "single-class nonfallback cells",int(((~paired.any_fallback)&~paired.auc_comparable).sum()),flush=True)
print(valid.groupby("design").agg(n=("delta_auc","size"),auc_wins=("delta_auc",lambda x:int((x>0).sum())),
    mean_delta_auc=("delta_auc","mean"),median_delta_auc=("delta_auc","median")).to_string(),flush=True)
print("DIAGNOSTIC ONLY; no preregistered confirmatory inference",flush=True)

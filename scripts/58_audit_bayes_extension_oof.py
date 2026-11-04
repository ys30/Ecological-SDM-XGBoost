"""Audit five-fold Bayesian extension artifacts against archived OOF baseline."""
from pathlib import Path
import pandas as pd
import numpy as np

root=Path("downloaded")
files=sorted(root.rglob("*_oof.csv"))
assert len(files)==10, f"Expected 10 OOF files, got {len(files)}: {files}"
ext=pd.concat((pd.read_csv(p) for p in files),ignore_index=True)
assert set(ext.variant)=={"quadratic","no_pool"}
assert ext.groupby(["variant","fold"]).size().shape[0]==10
assert not ext.duplicated(["variant","id"]).any()
base=pd.read_csv("results/bayes_spatial_cv_predictions.csv")
assert len(base)==9336
base=base.rename(columns={"mean_probability":"probability"})
base["variant"]="baseline"
cols=["id","plot","species","fold","observed","probability","variant"]
assert set(cols).issubset(ext.columns) and set(cols).issubset(base.columns)
allp=pd.concat([base[cols],ext[cols]],ignore_index=True)
for variant,g in allp.groupby("variant"):
    assert len(g)==9336, (variant,len(g))
    assert g.id.is_unique
    assert g[["id","plot","species","fold","observed"]].sort_values("id").reset_index(drop=True).equals(
        base[["id","plot","species","fold","observed"]].sort_values("id").reset_index(drop=True))
    assert g.probability.between(0,1).all()
def metrics(g):
    y=g.observed.to_numpy(); p=g.probability.to_numpy()
    positives=int(y.sum())
    auc=np.nan
    if 0<positives<len(y):
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(y,p))
    ap=np.nan
    if positives>0:
        order=np.argsort(-p,kind="stable")
        yy=y[order]
        ap=float((np.cumsum(yy)[yy==1]/(np.flatnonzero(yy==1)+1)).sum()/positives)
    return pd.Series(dict(n=len(y),positives=positives,auc=auc,ap=ap,
                          brier=float(np.mean((y-p)**2)),
                          logloss=float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12))+
                                                   (1-y)*np.log(np.clip(1-p,1e-12,1-1e-12))))))
m=allp.groupby(["variant","species","fold"],sort=True).apply(metrics,include_groups=False).reset_index()
out=Path("results/bayes_extensions");out.mkdir(parents=True,exist_ok=True)
m.to_csv(out/"five_fold_metrics.csv",index=False)
baseline=m[m.variant=="baseline"].drop(columns="variant")
joined=m[m.variant!="baseline"].merge(baseline,on=["species","fold"],suffixes=("","_baseline"),validate="one_to_one")
for metric in ["auc","ap","brier","logloss"]:
    joined[f"delta_{metric}"]=joined[metric]-joined[f"{metric}_baseline"]
joined.to_csv(out/"paired_baseline_differences.csv",index=False)
summary=joined.groupby("variant").agg(
    eligible_auc=("delta_auc","count"),
    mean_delta_auc=("delta_auc","mean"),
    mean_delta_ap=("delta_ap","mean"),
    mean_delta_brier=("delta_brier","mean"),
    mean_delta_logloss=("delta_logloss","mean")).reset_index()
summary.to_csv(out/"summary.csv",index=False)
with open(out/"SUMMARY.md","w") as f:
    f.write("# Bayesian extension five-fold results (archived short-chain baseline)\n\n")
    f.write("All 9,336 OOF rows per variant matched on ID, species, plot, fold and observation. Differences are exploratory and fold/species cells are correlated. AUC/AP omit undefined zero-positive cells; Brier/log loss include all cells. Baseline is archived shorter-chain predictions, not four-chain confirmation.\n\n")
    f.write(summary.to_markdown(index=False,floatfmt=".6f")+"\n")
print(summary.to_string(index=False))

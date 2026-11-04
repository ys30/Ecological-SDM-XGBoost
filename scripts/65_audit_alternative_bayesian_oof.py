"""Audit matched linear and quadratic Bayesian OOF predictions on alternative spatial folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
root=Path("downloaded")
files=sorted(root.rglob("*_oof.csv"))
assert len(files)==10, (len(files),[str(x) for x in files])
d=pd.concat((pd.read_csv(p) for p in files),ignore_index=True)
assert set(d.variant)=={"linear","quadratic"}
assert d.groupby(["variant","fold"]).size().shape[0]==10
assign=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
lookup=assign.set_index("plot").fold
for variant,g in d.groupby("variant"):
    assert len(g)==9336 and g.id.is_unique
    assert (g.fold.to_numpy()==g["plot"].map(lookup).to_numpy()).all()
    assert g.probability.between(0,1).all()
a=d[d.variant=="linear"].sort_values("id").reset_index(drop=True)
b=d[d.variant=="quadratic"].sort_values("id").reset_index(drop=True)
assert a[["id","plot","species","fold","observed"]].equals(b[["id","plot","species","fold","observed"]])
def metrics(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    positives=int(y.sum())
    auc=float(roc_auc_score(y,p)) if 0<positives<len(y) else np.nan
    ap=np.nan
    if positives:
        order=np.argsort(-p,kind="stable");yy=y[order]
        ap=float((np.cumsum(yy)[yy==1]/(np.flatnonzero(yy==1)+1)).sum()/positives)
    return pd.Series(dict(n=len(g),positives=positives,auc=auc,ap=ap,
        brier=float(np.mean((y-p)**2)),
        logloss=float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-p,1e-12,1-1e-12))))))
m=d.groupby(["variant","species","fold"],sort=True).apply(metrics,include_groups=False).reset_index()
base=m[m.variant=="linear"].drop(columns="variant")
paired=m[m.variant=="quadratic"].merge(base,on=["species","fold"],suffixes=("","_linear"),validate="one_to_one")
for k in ["auc","ap","brier","logloss"]:
    paired["delta_"+k]=paired[k]-paired[k+"_linear"]
summary=pd.DataFrame([dict(eligible_auc=int(paired.delta_auc.notna().sum()),
    eligible_ap=int(paired.delta_ap.notna().sum()),
    mean_delta_auc=paired.delta_auc.mean(),mean_delta_ap=paired.delta_ap.mean(),
    mean_delta_brier=paired.delta_brier.mean(),mean_delta_logloss=paired.delta_logloss.mean(),
    improved_auc=int((paired.delta_auc>0).sum()),improved_ap=int((paired.delta_ap>0).sum()),
    improved_brier=int((paired.delta_brier<0).sum()),
    improved_logloss=int((paired.delta_logloss<0).sum()))])
pooled=d.groupby("variant").apply(metrics,include_groups=False).reset_index()
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
m.to_csv(out/"alternative_taxon_fold_metrics.csv",index=False)
paired.to_csv(out/"alternative_paired_differences.csv",index=False)
summary.to_csv(out/"alternative_summary.csv",index=False)
pooled.to_csv(out/"alternative_pooled_metrics.csv",index=False)
with open(out/"ALTERNATIVE_BAYESIAN_VERIFIED.md","w") as f:
    f.write("# Alternative spatial partition: Bayesian linear versus quadratic\n\n")
    f.write("Five geographically clustered folds (15 k-means spatial clusters). Both variants were independently refitted using identical four-chain MCMC settings; 9,336 matched out-of-fold observations each. This is a **different within-dataset fold geometry**, not an independent external dataset.\n\n")
    f.write("## Taxon-fold paired summary\n\n"+summary.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("## Pooled OOF metrics\n\n"+pooled.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("AUC undefined in taxon-fold cells with only one class; AP undefined without positives. Fold/species cells are correlated. No formal inferential or multiplicity-adjusted claim.\n")
print("PAIRED SUMMARY");print(summary.to_string(index=False))
print("POOLED");print(pooled.to_string(index=False))

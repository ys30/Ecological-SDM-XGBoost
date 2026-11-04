"""Taxon-cluster uncertainty for matched four-model alternative-fold predictions.

Resample the eight taxa as clusters (not independent taxon-fold cells).
Report descriptive percentile intervals, not hypothesis-test p-values.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
gf=list(Path("glm").rglob("glm_linear_quadratic_oof.csv"))
assert len(gf)==1
d=pd.read_csv(gf[0])
bf=sorted(Path("bayes").rglob("*_oof.csv"))
assert len(bf)==10
b=pd.concat([pd.read_csv(p) for p in bf],ignore_index=True)
b["variant"]="bayesian_"+b.variant
d=pd.concat([d,b],ignore_index=True)
assert d.groupby("variant").size().eq(9336).all()
def metric(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    return pd.Series(dict(auc=roc_auc_score(y,p) if y.min()!=y.max() else np.nan,
        ap=average_precision_score(y,p) if y.sum()>0 else np.nan,
        brier=np.mean((y-p)**2),logloss=log_loss(y,p,labels=[0,1])))
cells=d.groupby(["variant","species","fold"]).apply(metric,include_groups=False).reset_index()
taxa=sorted(cells.species.unique())
assert len(taxa)==8
contrasts=[("glm_quadratic","bayesian_quadratic"),("glm_linear","bayesian_linear"),
           ("glm_quadratic","glm_linear"),("bayesian_quadratic","bayesian_linear")]
rows=[]
rng=np.random.default_rng(20261104)
for a,b in contrasts:
    x=cells[cells.variant==a].drop(columns="variant").merge(
        cells[cells.variant==b].drop(columns="variant"),on=["species","fold"],suffixes=("_a","_b"),validate="one_to_one")
    for k in ["auc","ap","brier","logloss"]:
        x["delta"]=x[k+"_a"]-x[k+"_b"]
        by=x.groupby("species").delta.mean()
        by=by.dropna()
        vals=by.to_numpy()
        draws=rng.choice(vals,size=(20000,len(vals)),replace=True).mean(axis=1)
        # Equal-taxon weighting: each taxon has equal influence, regardless of eligible cells.
        rows.append(dict(model_a=a,model_b=b,metric=k,n_taxa=len(vals),
            equal_taxon_mean=float(vals.mean()),bootstrap_ci_low=float(np.quantile(draws,.025)),
            bootstrap_ci_high=float(np.quantile(draws,.975)),
            n_taxa_favoring_a=int((vals>0).sum()) if k in ["auc","ap"] else int((vals<0).sum())))
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
res=pd.DataFrame(rows);res.to_csv(out/"four_model_taxon_cluster_bootstrap.csv",index=False)
print(res.to_string(index=False))
with open(out/"FOUR_MODEL_TAXON_CLUSTER_UNCERTAINTY.md","w") as f:
    f.write("# Four-model paired uncertainty across eight taxa\n\n")
    f.write("Descriptive 20,000-resample percentile bootstrap, resampling **taxa** as clusters and averaging each taxon's available fold-level metric differences before resampling. This yields an **equal-taxon estimand**, not the pooled OOF metric or equal-cell mean. AUC/AP undefined for one-class/no-positive taxon-fold cells. Only eight taxa and one dataset; intervals are exploratory, may be unstable, and do not account for shared geographic dependence between taxa or repeated fold geometries. No confirmatory p-values or external-validation claims.\n\n")
    f.write(res.to_markdown(index=False,floatfmt=".6f")+"\n")

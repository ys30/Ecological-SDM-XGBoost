"""Influence of each taxon on Bayesian extension paired OOF metrics.
Reads verified archived baseline and ten full-fold extension artifacts.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
files=sorted(Path("downloaded").rglob("*_oof.csv"))
assert len(files)==10
ext=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
base=pd.read_csv("results/bayes_spatial_cv_predictions.csv").rename(columns={"mean_probability":"probability"})
base["variant"]="baseline"
cols=["id","species","fold","observed","probability","variant"]
df=pd.concat([base[cols],ext[cols]],ignore_index=True)
assert df.groupby("variant").size().eq(9336).all()
out=Path("results/bayes_extensions");out.mkdir(parents=True,exist_ok=True)
def metric(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    auc=float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan
    ap=np.nan
    if y.sum()>0:
        order=np.argsort(-p,kind="stable")
        yy=y[order]
        ap=float((np.cumsum(yy)[yy==1]/(np.flatnonzero(yy==1)+1)).sum()/y.sum())
    return dict(auc=auc,ap=ap,brier=float(np.mean((y-p)**2)),
                logloss=float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-p,1e-12,1-1e-12)))))
# Equal-weight species/fold cells, matching original reported estimand
cells=[]
for (v,s,f),g in df.groupby(["variant","species","fold"]):
    cells.append(dict(variant=v,species=s,fold=f,**metric(g)))
m=pd.DataFrame(cells)
rows=[]
taxa=sorted(m.species.unique())
for excluded in ["none"]+taxa:
    g=m if excluded=="none" else m[m.species!=excluded]
    for variant in ["quadratic","no_pool"]:
        a=g[g.variant==variant].set_index(["species","fold"])
        b=g[g.variant=="baseline"].set_index(["species","fold"])
        assert a.index.equals(b.index)
        row=dict(excluded_taxon=excluded,variant=variant,n_taxa=g.species.nunique())
        for metric_name in ["auc","ap","brier","logloss"]:
            row["mean_delta_"+metric_name]=float((a[metric_name]-b[metric_name]).mean())
        rows.append(row)
r=pd.DataFrame(rows);r.to_csv(out/"leave_one_taxon_out.csv",index=False)
# Stronger audit: quantify how much quadratic gain depends on nsw18
quad=r[r.variant=="quadratic"]
with open(out/"TAXON_INFLUENCE_REPORT.md","w") as f:
    f.write("# Leave-one-taxon-out Bayesian sensitivity\n\n")
    f.write("Mean paired taxon-fold metric differences against the archived short-chain hierarchical baseline. Excluding each taxon in turn is a descriptive influence diagnostic, not a new external validation or independent hypothesis test. AUC/AP exclude undefined cells.\n\n")
    f.write(quad.to_markdown(index=False,floatfmt=".6f"))
    f.write("\n\n## No-pooling comparator\n\n")
    f.write(r[r.variant=="no_pool"].to_markdown(index=False,floatfmt=".6f"))
    f.write("\n")
print(r.to_string(index=False))

"""Recompute Bayesian extension deltas on exactly the archived three-pipeline comparison cells."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
files=sorted(Path("downloaded").rglob("*_oof.csv"))
assert len(files)==10
ext=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
base=pd.read_csv("results/bayes_spatial_cv_predictions.csv").rename(columns={"mean_probability":"probability"})
base["variant"]="baseline"
matched=pd.read_csv("results/matched_bayesian_maxent_xgboost_folds.csv")
keys=matched[["species","fold"]].drop_duplicates()
assert len(keys)==23, f"Expected 23 original matched cells, found {len(keys)}"
cols=["id","species","fold","observed","probability","variant"]
df=pd.concat([base[cols],ext[cols]],ignore_index=True).merge(keys,on=["species","fold"],validate="many_to_one")
rows=[]
for (v,s,f),g in df.groupby(["variant","species","fold"]):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    assert 0<y.sum()<len(y), (v,s,f)
    order=np.argsort(-p,kind="stable"); yy=y[order]
    rows.append(dict(variant=v,species=s,fold=f,
        auc=float(roc_auc_score(y,p)),
        ap=float((np.cumsum(yy)[yy==1]/(np.flatnonzero(yy==1)+1)).sum()/y.sum()),
        brier=float(np.mean((y-p)**2))))
m=pd.DataFrame(rows)
assert m.groupby("variant").size().eq(23).all()
b=m[m.variant=="baseline"].drop(columns="variant")
paired=m[m.variant!="baseline"].merge(b,on=["species","fold"],suffixes=("","_baseline"),validate="many_to_one")
for name in ["auc","ap","brier"]:
    paired["delta_"+name]=paired[name]-paired[name+"_baseline"]
summary=paired.groupby("variant").agg(n_cells=("delta_auc","count"),mean_delta_auc=("delta_auc","mean"),
    mean_delta_ap=("delta_ap","mean"),mean_delta_brier=("delta_brier","mean")).reset_index()
out=Path("results/bayes_extensions");out.mkdir(parents=True,exist_ok=True)
paired.to_csv(out/"original_23_cells_paired.csv",index=False)
summary.to_csv(out/"original_23_cells_summary.csv",index=False)
with open(out/"ORIGINAL_23_CELLS_REPORT.md","w") as f:
    f.write("# Bayesian extensions on the original 23 three-pipeline matched cells\n\n")
    f.write("These 23 taxon-fold cells are the exact keys from results/matched_bayesian_maxent_xgboost_folds.csv. All three Bayesian variants use identical held-out records within each cell. This is the appropriate AUC/AP subset for direct comparison with the original three-pipeline summary; it is distinct from the extension-only 30-cell analysis. Baseline predictions are archived shorter-chain results.\n\n")
    f.write(summary.to_markdown(index=False,floatfmt=".6f")+"\n")
print(summary.to_string(index=False))

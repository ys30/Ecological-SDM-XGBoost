"""Compare GLM and Bayesian predictions on identical alternative-spatial OOF records."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
gfiles=list(Path("glm").rglob("glm_linear_quadratic_oof.csv"))
assert len(gfiles)==1,gfiles
glm=pd.read_csv(gfiles[0]); bfiles=sorted(Path("bayes").rglob("*_oof.csv"))
assert len(bfiles)==10,len(bfiles)
bayes=pd.concat([pd.read_csv(p) for p in bfiles],ignore_index=True)
bayes.variant="bayesian_"+bayes.variant
d=pd.concat([glm,bayes],ignore_index=True)
assert d.groupby("variant").size().eq(9336).all()
assert d.groupby("variant").id.nunique().eq(9336).all()
cols=["id","plot","species","fold","observed"]
ref=None
for variant,g in d.groupby("variant"):
    t=g[cols].sort_values("id").reset_index(drop=True)
    if ref is None:ref=t
    else:assert ref.equals(t),variant
def metrics(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    return pd.Series(dict(n=len(g),positives=int(y.sum()),auc=roc_auc_score(y,p) if len(set(y))==2 else np.nan,
        ap=average_precision_score(y,p) if sum(y)>0 else np.nan,brier=np.mean((y-p)**2),logloss=log_loss(y,p,labels=[0,1])))
m=d.groupby(["variant","species","fold"]).apply(metrics,include_groups=False).reset_index()
pooled=d.groupby("variant").apply(metrics,include_groups=False).reset_index()
pairs=[("glm_quadratic","bayesian_quadratic"),("glm_linear","bayesian_linear"),("glm_quadratic","glm_linear"),("bayesian_quadratic","bayesian_linear")]
rows=[]
for a,b in pairs:
    aa=m[m.variant==a].drop(columns="variant");bb=m[m.variant==b].drop(columns="variant")
    x=aa.merge(bb,on=["species","fold"],suffixes=("_a","_b"),validate="one_to_one")
    row=dict(model_a=a,model_b=b)
    for k in ["auc","ap","brier","logloss"]:
        z=x[k+"_a"]-x[k+"_b"];row["mean_cell_delta_"+k]=z.mean()
        row["eligible_"+k]=z.notna().sum()
    rows.append(row)
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
pooled.to_csv(out/"four_model_pooled.csv",index=False)
m.to_csv(out/"four_model_taxon_fold.csv",index=False)
paired=pd.DataFrame(rows);paired.to_csv(out/"four_model_paired_differences.csv",index=False)
print("POOLED\n"+pooled.to_string(index=False))
print("PAIRED\n"+paired.to_string(index=False))
with open(out/"FOUR_MODEL_COMPARISON_VERIFIED.md","w") as f:
    f.write("# Four-model comparison on matched alternative-spatial OOF records\n\n")
    f.write("All four variants have exactly matched 9,336 plot–taxon OOF observations and 747 positives. GLMs use per-taxon L2-regularized logistic regression; Bayesian variants use hierarchical brms. All models trained within the same alternative geographic five-fold scheme. **Within-dataset sensitivity only; not external validation.**\n\n")
    f.write("## Pooled OOF metrics\n\n"+pooled.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("## Mean taxon-fold differences: model A minus B\n\n"+paired.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("AUC/AP may be undefined for one-class cells. These descriptive differences do not constitute inferential tests; model families differ in shrinkage, pooling and fitting, not only algorithm name.\n")

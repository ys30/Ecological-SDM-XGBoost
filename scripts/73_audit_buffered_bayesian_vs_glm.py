"""Audit matched Bayesian and GLM buffered out-of-fold predictions.

Run after all 30 buffered Bayesian jobs have completed and their artifacts
have been downloaded into results/bayes_buffered/{5,10,20}km.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
base=Path("results")
g=pd.read_csv(base/"alternative_spatial_validation/buffered_glm/buffered_glm_oof.csv")
assert set(g.buffer_km)=={0,5,10,20}
rows=[];paired=[]
for radius in (5,10,20):
 for variant in ("linear","quadratic"):
  files=[base/f"bayes_buffered/{radius}km/{variant}_fold{i}_oof.csv" for i in range(1,6)]
  missing=[str(f) for f in files if not f.exists()]
  if missing: raise FileNotFoundError("Missing Bayesian fold outputs: "+", ".join(missing))
  b=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
  a=g[(g.buffer_km==radius)&(g.variant=="glm_"+variant)].copy()
  assert len(a)==len(b)==9336 and a.id.is_unique and b.id.is_unique
  merged=a.merge(b,on="id",suffixes=("_glm","_bayes"),validate="one_to_one")
  assert len(merged)==9336
  for x,y in [("species","species"),("fold","fold"),("observed","observed")]:
   assert (merged[x+"_glm"].astype(str)==merged[y+"_bayes"].astype(str)).all(),x
  def metrics(y,p):
   return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),
      brier=float(np.mean((y-p)**2)),logloss=log_loss(y,p,labels=[0,1]))
  for model,col in (("glm","probability_glm"),("bayesian","probability_bayes")):
   m=metrics(merged.observed_glm.to_numpy(),merged[col].to_numpy())
   rows.append(dict(buffer_km=radius,variant=variant,model=model,n=len(merged),**m))
  for (species,fold),sub in merged.groupby(["species_glm","fold_glm"]):
   y=sub.observed_glm.to_numpy()
   for metric in ("auc","ap","brier","logloss"):
    if metric=="auc" and len(set(y))<2: continue
    if metric=="ap" and y.sum()==0: continue
    m1=metrics(y,sub.probability_glm.to_numpy())[metric] if len(set(y))==2 else (
       float(np.mean((y-sub.probability_glm.to_numpy())**2)) if metric=="brier" else log_loss(y,sub.probability_glm,labels=[0,1]))
    m2=metrics(y,sub.probability_bayes.to_numpy())[metric] if len(set(y))==2 else (
       float(np.mean((y-sub.probability_bayes.to_numpy())**2)) if metric=="brier" else log_loss(y,sub.probability_bayes,labels=[0,1]))
    paired.append(dict(buffer_km=radius,variant=variant,species=species,fold=fold,metric=metric,glm_minus_bayesian=m1-m2))
out=base/"alternative_spatial_validation/buffered_four_model"
out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/"pooled_metrics.csv",index=False)
pd.DataFrame(paired).to_csv(out/"paired_taxon_fold_differences.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(paired).groupby(["buffer_km","variant","metric"]).glm_minus_bayesian.agg(["mean","count"]).to_string())

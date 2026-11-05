"""Aggregate rare-taxon positive-record probability/rank diagnostics, no row-level release."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
root=Path("results")
g=pd.read_csv(root/"alternative_spatial_validation/buffered_glm/buffered_glm_oof.csv")
rows=[]
for km in (5,10,20):
 for fold in range(1,6):
  b=pd.read_csv(root/f"bayes_buffered/{km}km/quadratic_fold{fold}_oof.csv")
  a=g[(g.buffer_km==km)&(g.variant=="glm_quadratic")&(g.fold==fold)]
  m=a.merge(b,on="id",suffixes=("_glm","_bayes"),validate="one_to_one")
  assert len(m)==len(a)==len(b)
  assert (m.observed_glm.to_numpy()==m.observed_bayes.to_numpy()).all()
  assert (m.species_glm.astype(str).to_numpy()==m.species_bayes.astype(str).to_numpy()).all()
  for taxon in ("nsw19","nsw20","nsw22"):
   t=m[m.species_glm==taxon].copy()
   assert len(t)>0
   pos=t[t.observed_glm==1]
   for model in ("glm","bayes"):
    p=t[f"probability_{model}"].to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    ranks=rankdata(-p,method="average")
    mask=t.observed_glm.to_numpy()==1
    rows.append(dict(buffer_km=km,fold=fold,species=taxon,model=model,
      n_test=len(t),n_positive=int(mask.sum()),
      positive_mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
      positive_mean_rank=float(ranks[mask].mean()) if mask.any() else np.nan,
      positive_mean_logloss=float(-np.log(np.clip(p[mask],1e-15,1)).mean()) if mask.any() else np.nan,
      negative_mean_probability=float(p[~mask].mean()) if (~mask).any() else np.nan))
out=Path("results/alternative_spatial_validation/rare_record_diagnostics")
out.mkdir(parents=True,exist_ok=True)
d=pd.DataFrame(rows)
d.to_csv(out/"rare_taxon_fold_aggregate_probability_rank.csv",index=False)
p=d.pivot(index=["buffer_km","fold","species","n_positive"],columns="model",values=["positive_mean_probability","positive_mean_rank","positive_mean_logloss","negative_mean_probability"])
p.columns=[f"{a}_{b}" for a,b in p.columns]
p=p.reset_index()
p["positive_logloss_glm_minus_bayes"]=p.positive_mean_logloss_glm-p.positive_mean_logloss_bayes
p["positive_rank_glm_minus_bayes"]=p.positive_mean_rank_glm-p.positive_mean_rank_bayes
p.to_csv(out/"rare_taxon_fold_matched_diagnostics.csv",index=False)
print("10 KM POSITIVE RECORD AGGREGATES (no row-level data)")
print(p[p.buffer_km==10].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

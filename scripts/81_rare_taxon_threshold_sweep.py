"""Rare-taxon threshold sweep on matched buffered OOF predictions; aggregate only."""
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results")
g=pd.read_csv(root/"alternative_spatial_validation/buffered_glm/buffered_glm_oof.csv")
thresholds=[0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5]
rows=[]
for km in (5,10,20):
 for fold in range(1,6):
  b=pd.read_csv(root/f"bayes_buffered/{km}km/quadratic_fold{fold}_oof.csv")
  a=g[(g.buffer_km==km)&(g.variant=="glm_quadratic")&(g.fold==fold)]
  m=a.merge(b,on="id",suffixes=("_glm","_bayes"),validate="one_to_one")
  assert len(m)==len(a)==len(b)
  assert (m.observed_glm.to_numpy()==m.observed_bayes.to_numpy()).all()
  for taxon in ("nsw19","nsw20","nsw22"):
   t=m[m.species_glm==taxon]
   y=t.observed_glm.to_numpy(dtype=int)
   for model in ("glm","bayes"):
    p=t[f"probability_{model}"].to_numpy(float)
    for th in thresholds:
     sel=p>=th
     rows.append(dict(buffer_km=km,fold=fold,species=taxon,model=model,threshold=th,
       tp=int((sel&(y==1)).sum()),fp=int((sel&(y==0)).sum()),
       fn=int((~sel&(y==1)).sum()),tn=int((~sel&(y==0)).sum())))
d=pd.DataFrame(rows)
out=root/"alternative_spatial_validation/rare_threshold_sweep"
out.mkdir(parents=True,exist_ok=True)
d.to_csv(out/"fold_threshold_counts.csv",index=False)
z=d.groupby(["buffer_km","species","model","threshold"],as_index=False)[["tp","fp","fn","tn"]].sum()
z["precision"]=z.tp/(z.tp+z.fp).replace(0,np.nan)
z["recall"]=z.tp/(z.tp+z.fn)
z["fpr"]=z.fp/(z.fp+z.tn)
z.to_csv(out/"pooled_threshold_metrics.csv",index=False)
print("10 KM THRESHOLD SWEEP")
print(z[z.buffer_km==10].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

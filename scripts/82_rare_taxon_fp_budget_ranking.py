"""Matched rare-taxon OOF ranking at fixed false-positive budgets; aggregate only."""
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results")
g=pd.read_csv(root/"alternative_spatial_validation/buffered_glm/buffered_glm_oof.csv")
budgets=(0,5,10,20,50,100)
rows=[]
for km in (5,10,20):
 blocks=[]
 for fold in range(1,6):
  b=pd.read_csv(root/f"bayes_buffered/{km}km/quadratic_fold{fold}_oof.csv")
  a=g[(g.buffer_km==km)&(g.variant=="glm_quadratic")&(g.fold==fold)]
  m=a.merge(b,on="id",suffixes=("_glm","_bayes"),validate="one_to_one")
  assert len(m)==len(a)==len(b)
  assert (m.observed_glm.to_numpy()==m.observed_bayes.to_numpy()).all()
  assert (m.species_glm.astype(str).to_numpy()==m.species_bayes.astype(str).to_numpy()).all()
  blocks.append(m)
 d=pd.concat(blocks,ignore_index=True)
 for taxon in ("nsw19","nsw20","nsw22"):
  t=d[d.species_glm==taxon]
  y=t.observed_glm.to_numpy(dtype=int)
  assert len(t)==1167
  for model in ("glm","bayes"):
   p=t[f"probability_{model}"].to_numpy(float)
   assert np.isfinite(p).all()
   order=np.argsort(-p,kind="stable")
   ys=y[order]
   cumulative_fp=np.cumsum(ys==0)
   for budget in budgets:
    # Diagnostic oracle: includes every prefix with <= budget false positives.
    # For a tie at boundary, deterministic order is used; report ties separately.
    eligible=np.flatnonzero(cumulative_fp<=budget)
    k=int(eligible[-1]+1) if len(eligible) else 0
    tp=int(ys[:k].sum())
    fp=int(k-tp)
    rows.append(dict(buffer_km=km,species=taxon,model=model,fp_budget=budget,
      n= len(t),positives=int(y.sum()),selected=k,tp=tp,fp=fp,
      recall=tp/int(y.sum()),precision=tp/k if k else np.nan,
      boundary_score=float(p[order[k-1]]) if k else np.nan,
      boundary_tie_count=int(np.count_nonzero(p==p[order[k-1]])) if k else 0))
out=root/"alternative_spatial_validation/rare_fp_budget"
out.mkdir(parents=True,exist_ok=True)
z=pd.DataFrame(rows)
z.to_csv(out/"rare_taxon_oracle_fp_budget_metrics.csv",index=False)
print("10 KM FALSE-POSITIVE-BUDGET ORACLE DIAGNOSTIC")
print(z[z.buffer_km==10].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("WARNING: FP budgets use test labels to locate cutoff, diagnostic only, NOT deployable threshold selection.")

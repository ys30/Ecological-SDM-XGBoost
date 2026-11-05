"""Rare-taxon matched OOF calibration/error audit; publish aggregates only."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
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
  for species in ("nsw19","nsw20","nsw22"):
   t=m[m.species_glm==species]
   y=t.observed_glm.to_numpy(dtype=int)
   for model in ("glm","bayes"):
    p=t[f"probability_{model}"].to_numpy(dtype=float)
    assert len(p)==len(y) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    rows.append(dict(buffer_km=km,fold=fold,species=species,model=model,
      n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
      mean_probability=float(p.mean()),mean_probability_minus_prevalence=float(p.mean()-y.mean()),
      brier=float(brier_score_loss(y,p)),
      logloss=float(log_loss(y,p,labels=[0,1])),
      high_risk_count=int((p>=0.1).sum()),high_risk_false_positives=int(((p>=0.1)&(y==0)).sum()),
      high_risk_true_positives=int(((p>=0.1)&(y==1)).sum())))
d=pd.DataFrame(rows)
out=root/"alternative_spatial_validation/rare_calibration_diagnostics"
out.mkdir(parents=True,exist_ok=True)
d.to_csv(out/"fold_calibration_error_aggregates.csv",index=False)
# Pool predictions using fold-size weighting for proper scoring rules.
s=(d.groupby(["buffer_km","species","model"],as_index=False)
 .apply(lambda x:pd.Series(dict(n=int(x.n.sum()),positives=int(x.positives.sum()),
 mean_probability=float(np.average(x.mean_probability,weights=x.n)),
 prevalence=float(x.positives.sum()/x.n.sum()),
 brier=float(np.average(x.brier,weights=x.n)),
 logloss=float(np.average(x.logloss,weights=x.n)),
 high_risk_count=int(x.high_risk_count.sum()),
 high_risk_true_positives=int(x.high_risk_true_positives.sum()),
 high_risk_false_positives=int(x.high_risk_false_positives.sum()))),include_groups=False))
s.to_csv(out/"pooled_calibration_error_aggregates.csv",index=False)
print("10 KM RARE-TAXON PROBABILITY/ERROR AUDIT")
print(s[s.buffer_km==10].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
print("Note: prevalence versus mean probability is calibration-in-the-large, not a reliability curve.")

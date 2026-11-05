"""Aggregate-only covariate shift audit for nested spatial GLM outer folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
p=d[["plot","easting","northing","rainann","tempann","soildepth","topo"]].drop_duplicates("plot").merge(a[["plot","fold"]],on="plot",validate="one_to_one")
assert len(p)==1167 and p["plot"].is_unique
f=["rainann","tempann","soildepth","topo"]
rows=[]
for km in (5,10,20):
 for fold in range(1,6):
  test=p[p.fold==fold]
  train=p[p.fold!=fold]
  rawdist=cKDTree(test[["easting","northing"]].to_numpy()).query(train[["easting","northing"]].to_numpy())[0]
  train=train.loc[rawdist>=1000*km]
  assert len(train)>100
  nearest=cKDTree(train[["easting","northing"]].to_numpy()).query(test[["easting","northing"]].to_numpy())[0]
  for feature in f:
   x=train[feature].to_numpy(float);y=test[feature].to_numpy(float)
   assert np.isfinite(x).all() and np.isfinite(y).all()
   sd=x.std(ddof=1)
   smd=(y.mean()-x.mean())/sd if sd>0 else np.nan
   rows.append(dict(buffer_km=km,outer_fold=fold,feature=feature,n_train=len(train),n_test=len(test),
     standardized_mean_difference=smd,absolute_smd=abs(smd),
     train_mean=x.mean(),test_mean=y.mean(),train_sd=sd,
     test_outside_train_range_fraction=np.mean((y<x.min())|(y>x.max())),
     nearest_train_xy_median_units=np.median(nearest),
     nearest_train_xy_p90_units=np.quantile(nearest,.9)))
z=pd.DataFrame(rows)
out=Path("results/alternative_spatial_validation/covariate_shift_audit")
out.mkdir(parents=True,exist_ok=True)
z.to_csv(out/"outer_fold_covariate_shift.csv",index=False)
agg=z.groupby(["buffer_km","outer_fold"],as_index=False).agg(
 max_abs_smd=("absolute_smd","max"),
 mean_abs_smd=("absolute_smd","mean"),
 max_outside_fraction=("test_outside_train_range_fraction","max"),
 median_nearest_train_xy=("nearest_train_xy_median_units","first"),
 p90_nearest_train_xy=("nearest_train_xy_p90_units","first"))
agg.to_csv(out/"outer_fold_shift_summary.csv",index=False)
print("10 KM COVARIATE SHIFT SUMMARY")
print(agg[agg.buffer_km==10].to_string(index=False))
print("10 KM PER FEATURE")
print(z[z.buffer_km==10][["outer_fold","feature","standardized_mean_difference","test_outside_train_range_fraction"]].to_string(index=False))
print("Descriptive diagnostics only; no causal attribution or validated CRS. No model prediction probabilities were examined.")

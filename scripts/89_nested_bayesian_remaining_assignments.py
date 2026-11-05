"""Prepare nested Bayesian pilot fold assignments without outer-test leakage."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
p=d[["plot","easting","northing"]].drop_duplicates("plot").merge(a[["plot","fold"]],on="plot",validate="one_to_one")
import sys
km=10; fold=int(sys.argv[1])
assert fold in (1,2,4,5)
test=p[p.fold==fold];cand=p[p.fold!=fold].copy()
distance=cKDTree(test[["easting","northing"]]).query(cand[["easting","northing"]])[0]
cand=cand.loc[distance>=km*1000].copy()
xy=cand[["easting","northing"]].to_numpy(float)
xy=(xy-xy.mean(0))/xy.std(0)
cand["inner_fold"]=KMeans(n_clusters=4,n_init=10,random_state=20261104+fold).fit_predict(xy)
rows=[]
for inner in range(4):
 val=cand[cand.inner_fold==inner]
 tr=cand[cand.inner_fold!=inner]
 dist=cKDTree(val[["easting","northing"]]).query(tr[["easting","northing"]])[0]
 tr=tr.loc[dist>=km*1000]
 assert len(tr)>0 and set(tr["plot"]).isdisjoint(set(val["plot"]))
 for x in tr["plot"]:rows.append((inner,"train",x))
 for x in val["plot"]:rows.append((inner,"validation",x))
for x in cand["plot"]:rows.append((-1,"train",x))
for x in test["plot"]:rows.append((-1,"test",x))
out=Path("results/nested_bayesian_pilot")
out.mkdir(parents=True,exist_ok=True)
z=pd.DataFrame(rows,columns=["inner_fold","role","plot"])
z.to_csv(out/("fold%d_10km_roles.csv"%fold)",index=False)
print(z.groupby(["inner_fold","role"]).size().to_string())
print("Pilot: 4 inner hierarchical fits + 1 outer fit; outer-test labels not used for fitting or threshold selection.")

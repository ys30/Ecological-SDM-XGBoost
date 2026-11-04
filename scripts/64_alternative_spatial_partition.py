"""Generate an alternative, geographically coherent five-fold partition for SDM sensitivity.
Deterministic k-means spatial clusters, then balance cluster groups into five folds.
This does NOT constitute independent geographic validation.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
d=pd.read_csv("data/train.csv")
plots=d[["plot","easting","northing"]].drop_duplicates("plot").sort_values("plot").reset_index(drop=True)
assert len(plots)==1167 and plots[["easting","northing"]].notna().all().all()
xy=plots[["easting","northing"]].to_numpy(dtype=float)
xy=(xy-xy.mean(axis=0))/xy.std(axis=0)
km=KMeans(n_clusters=15,random_state=20261104,n_init=30)
plots["cluster"]=km.fit_predict(xy)
counts=plots.groupby("cluster").size().sort_values(ascending=False)
fold_sizes=np.zeros(5,dtype=int);assignment={}
for cluster,n in counts.items():
    fold=int(np.argmin(fold_sizes))
    assignment[int(cluster)]=fold+1
    fold_sizes[fold]+=int(n)
plots["fold"]=plots.cluster.map(assignment)
assert plots.fold.nunique()==5
assert plots.groupby("fold").size().min()>=100
assert plots.groupby("cluster").fold.nunique().max()==1
original=pd.read_csv("results/spatial_2d_block_assignments.csv")[["plot","fold"]].rename(columns={"fold":"original_fold"})
joined=plots.merge(original,on="plot",validate="one_to_one")
# Fold labels are arbitrary; use adjusted Rand index to measure distinct partition
from sklearn.metrics import adjusted_rand_score
ari=adjusted_rand_score(joined.original_fold,joined.fold)
assert ari<0.95, f"Alternative partition too similar to original: {ari}"
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
plots[["plot","cluster","fold"]].to_csv(out/"kmeans15_five_fold_assignments.csv",index=False)
summary=plots.groupby("fold").agg(plots=("plot","size"),clusters=("cluster","nunique")).reset_index()
summary.to_csv(out/"partition_summary.csv",index=False)
with open(out/"PARTITION_AUDIT.md","w") as f:
    f.write("# Alternative geographic five-fold partition audit\n\n")
    f.write("Deterministic 15-cluster k-means on standardized projected coordinates, greedy allocation of whole clusters to five folds by plot counts. Random seed 20261104; n_init=30. No species outcomes used in partition construction. All taxa from a plot stay in the same fold. This is a sensitivity to fold geometry, **not** an independent dataset or external validation.\n\n")
    f.write(f"Adjusted Rand index versus original fold labels: **{ari:.6f}** (lower indicates more distinct partitions).\n\n")
    f.write(summary.to_markdown(index=False)+"\n")
print(summary.to_string(index=False));print("Adjusted Rand index",ari)

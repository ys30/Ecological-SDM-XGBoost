"""Preregistered spatial inner folds: train-only kmeans, random-order greedy allocation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def spatial_inner_folds(frame, outer_test_plots, n_clusters=12, n_folds=4, seed=None, outer_fold=None):
    if seed is None:
        if outer_fold is None:
            raise ValueError("outer_fold or explicit seed required; preregistered seed=6220+outer_fold")
        seed=6220+int(outer_fold)
    needed={"plot","easting","northing"}
    if not needed <= set(frame):
        raise KeyError(f"missing columns: {needed-set(frame)}")
    if n_clusters<n_folds or n_folds<2:
        raise ValueError("invalid cluster/fold counts")
    coords=frame[["plot","easting","northing"]].copy()
    if coords.isna().any().any():
        raise ValueError("missing coordinates")
    if coords.groupby("plot")[["easting","northing"]].nunique().gt(1).any().any():
        raise ValueError("inconsistent coordinates for same plot")
    coords=coords.drop_duplicates("plot").sort_values("plot")
    outer=set(outer_test_plots)
    if not outer or not outer <= set(coords["plot"]):
        raise ValueError("invalid outer test plots")
    train=coords.loc[~coords["plot"].isin(outer)].copy()
    if len(train)<n_clusters:
        raise ValueError("too few training plots")
    xy=train[["easting","northing"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("nonfinite coordinates")
    km=KMeans(n_clusters=n_clusters,random_state=seed,n_init=20)
    train["cluster"]=km.fit_predict(xy)
    # Predeclared random-order greedy assignment; never sort clusters by size.
    counts=train.groupby("cluster").size().to_dict()
    order=np.random.default_rng(seed).permutation(sorted(counts))
    totals=[0]*n_folds
    cluster_to_fold={}
    for cluster in order:
        fold=min(range(n_folds),key=lambda f:(totals[f],f))
        cluster_to_fold[int(cluster)]=fold
        totals[fold]+=int(counts[int(cluster)])
    train["inner_fold"]=train["cluster"].map(cluster_to_fold)
    if train["inner_fold"].nunique()!=n_folds:
        raise ValueError("empty inner fold")
    return train[["plot","cluster","inner_fold"]].reset_index(drop=True)

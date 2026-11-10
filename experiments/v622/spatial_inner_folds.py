"""Deterministic, outer-training-only spatial k-means inner fold assignment.

This is a plot-level fold constructor, not a fitted model or performance test.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def spatial_inner_folds(frame, outer_test_plots, n_clusters=12, n_folds=4, seed=2026):
    needed={"plot","easting","northing"}
    if not needed <= set(frame):
        raise KeyError(f"missing columns: {needed-set(frame)}")
    if n_clusters<n_folds:
        raise ValueError("n_clusters must be >= n_folds")
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
    # Assign entire clusters to folds; deterministic greedy balancing by plot count.
    counts=train.groupby("cluster").size().sort_values(ascending=False,kind="stable")
    totals=[0]*n_folds
    cluster_to_fold={}
    for cluster,count in counts.items():
        fold=min(range(n_folds),key=lambda f:(totals[f],f))
        cluster_to_fold[int(cluster)]=fold
        totals[fold]+=int(count)
    train["inner_fold"]=train["cluster"].map(cluster_to_fold)
    if train["inner_fold"].nunique()!=n_folds:
        raise ValueError("empty inner fold")
    return train[["plot","cluster","inner_fold"]].reset_index(drop=True)

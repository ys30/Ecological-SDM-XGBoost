"""Synthetic nested spatial fold -> train-only design -> GLM integration smoke.
Engineering test only; no outer predictions, performance, or paper outputs.
"""
import numpy as np
import pandas as pd
from .spatial_inner_folds import spatial_inner_folds
from .design_matrix import prepare_design_fold
from .glm_adapter import run_glm_inner

def main():
    rng=np.random.default_rng(622)
    n=160
    plot=np.arange(n)
    frame=pd.DataFrame({
        "plot":plot,
        "easting":(plot%20)*1000.+rng.normal(0,50,n),
        "northing":(plot//20)*1000.+rng.normal(0,50,n),
        "rainann":rng.normal(100,20,n),
        "soildepth":rng.normal(30,5,n),
        "tempann":rng.normal(20,3,n),
        "topo":rng.normal(0,1,n),
    })
    frame["Species"]="synthetic"
    frame["pres.abs"]=rng.binomial(1,.35,n)
    outer=set(plot[::5])
    folds=spatial_inner_folds(frame,outer,outer_fold=1)
    assert set(folds["plot"])==set(plot)-outer
    assert folds.groupby("cluster")["inner_fold"].nunique().max()==1
    for k in range(4):
        validation=set(folds.loc[folds["inner_fold"]==k,"plot"])
        train,val,_,design=prepare_design_fold(frame,outer,validation,"glm_quad_rt",standardize=True)
        train_mask=~frame["plot"].isin(outer|validation)
        val_mask=frame["plot"].isin(validation)
        ytr=frame.loc[train_mask,"pres.abs"].to_numpy()
        yval=frame.loc[val_mask,"pres.abs"].to_numpy()
        result=run_glm_inner(ytr,yval,train.to_numpy(),val.to_numpy())
        assert len(result.predictions)==len(yval)
        assert np.isfinite(result.predictions).all()
        assert not set(frame.loc[train_mask,"plot"])&validation
        assert not set(frame.loc[train_mask,"plot"])&outer
        assert not validation&outer
        print(f"PASS synthetic spatial inner fold {k}: train={len(ytr)} validation={len(yval)} status={result.status}")
if __name__=="__main__":
    main()

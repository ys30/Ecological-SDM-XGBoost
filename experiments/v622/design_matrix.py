"""Train-only feature engineering for explicitly supported v6.22 GLM/GLMnet designs.

This builds matrices only; no model fitting, hyperparameter selection or outer scoring.
"""
import numpy as np
import pandas as pd
from .feature_contract import predictor_columns

QUADRATIC_RT={"glm_quad_rt","glm_shared_quad_rt","glmm_quad_rt","bayes_quad","glmnet_quad_rt"}
QUADRATIC_ALL={"glm_quad_env","glmnet_quad_env"}
LINEAR={"glm_lin","glm_shared_lin","glmm_lin","bayes_lin","glmnet_lin"}
SUPPORTED=QUADRATIC_RT|QUADRATIC_ALL|LINEAR
BASE=("rainann","soildepth","tempann","topo")

class FoldDesign:
    def __init__(self,model_id,standardize=False):
        if model_id not in SUPPORTED:
            raise ValueError("unsupported feature-design family; use separate model-specific implementation")
        self.model_id=model_id
        self.standardize=bool(standardize)
        self.numeric,self.categorical=predictor_columns(model_id)
    def fit(self,train):
        if len(train)==0:
            raise ValueError("empty training partition")
        self.median_={}
        self.mean_={}
        self.scale_={}
        for name in self.numeric:
            vals=pd.to_numeric(train[name],errors="coerce").replace([np.inf,-np.inf],np.nan)
            med=float(vals.median())
            if not np.isfinite(med):
                raise ValueError("no finite training data: "+name)
            clean=vals.fillna(med).to_numpy(float)
            self.median_[name]=med
            self.mean_[name]=float(clean.mean())
            std=float(clean.std(ddof=1)) if len(clean)>1 else 0.0
            self.scale_[name]=std if std>0 else 1.
        self.levels_={c:tuple(sorted(str(x) for x in train[c].dropna().unique())) for c in self.categorical}
        self.columns_=list(BASE)
        if self.model_id in QUADRATIC_RT:
            self.columns_+=["rainann_sq","tempann_sq"]
        elif self.model_id in QUADRATIC_ALL:
            self.columns_ += [f"{c}_sq" for c in BASE]
        self.columns_ += [f"{c}={v}" for c in self.categorical for v in self.levels_[c]]
        return self
    def transform(self,frame):
        if not hasattr(self,"columns_"):
            raise RuntimeError("fit on training partition first")
        out={}
        for c in BASE:
            vals=pd.to_numeric(frame[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(self.median_[c]).to_numpy(float)
            if self.standardize:
                vals=(vals-self.mean_[c])/self.scale_[c]
            out[c]=vals
        if self.model_id in QUADRATIC_RT:
            squared=("rainann","tempann")
        elif self.model_id in QUADRATIC_ALL:
            squared=BASE
        else:
            squared=()
        for c in squared:
            out[f"{c}_sq"]=out[c]**2
        for c in self.categorical:
            values=frame[c].astype("string")
            for v in self.levels_[c]:
                out[f"{c}={v}"]=values.eq(v).fillna(False).to_numpy(float)
        return pd.DataFrame(out,index=frame.index).loc[:,self.columns_]
    def fit_transform(self,train):
        return self.fit(train).transform(train)

def prepare_design_fold(frame,outer_test_plots,inner_validation_plots,model_id,standardize=False):
    outer=set(outer_test_plots)
    inner=set(inner_validation_plots)
    plots=set(frame["plot"])
    if not outer or not inner or outer&inner or not outer<=plots or not inner<=plots:
        raise ValueError("invalid or overlapping partitions")
    tr=frame.loc[~frame["plot"].isin(outer|inner)]
    val=frame.loc[frame["plot"].isin(inner)]
    test=frame.loc[frame["plot"].isin(outer)]
    design=FoldDesign(model_id,standardize=standardize).fit(tr)
    return design.transform(tr),design.transform(val),design.transform(test),design

"""Engineering-only logistic GLM adapter for EMS v6.22.

Statsmodels binomial GLM; explicit intercept; no tuning, no outer test scoring.
Primary: Newton maxiter=100; retry: default IRLS maxiter=200.
This is an implementation candidate, NOT verified equivalence to historical R glm.
"""
import numpy as np
import statsmodels.api as sm
from .fit_retry_policy import run_with_retry

def _fit_predict(train_x,y_train,validation_x,method,maxiter):
    x=np.asarray(train_x,dtype=float)
    v=np.asarray(validation_x,dtype=float)
    y=np.asarray(y_train,dtype=float)
    if x.ndim!=2 or v.ndim!=2 or x.shape[1]!=v.shape[1]:
        raise ValueError("incompatible design matrices")
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite predictors")
    x=sm.add_constant(x,has_constant="add")
    v=sm.add_constant(v,has_constant="add")
    model=sm.GLM(y,x,family=sm.families.Binomial())
    result=model.fit(method=method,maxiter=maxiter,disp=0)
    if not bool(result.converged):
        raise RuntimeError("GLM did not converge")
    return np.asarray(result.predict(v),dtype=float)

def run_glm_inner(y_train,y_validation,train_x,validation_x):
    return run_with_retry(
        y_train,y_validation,train_x,validation_x,
        fit_primary=lambda x,y,v:_fit_predict(x,y,v,"newton",100),
        fit_retry=lambda x,y,v:_fit_predict(x,y,v,"IRLS",200))

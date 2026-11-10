"""Pre-freeze sklearn elastic-net logistic engineering adapter.

Not R glmnet equivalent. No lambda-path generation or nested tuning here.
Input design must already be standardized on training-only moments.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .fit_retry_policy import run_with_retry

def _predict(x,y,v,alpha,lam,max_iter):
    if alpha not in (0.0,0.5,1.0):
        raise ValueError("alpha must be one of the preregistered values")
    if not np.isfinite(lam) or lam<=0:
        raise ValueError("lambda must be positive")
    # sklearn objective uses 1/(2*C) for L2 vs glmnet lambda convention;
    # C=1/lambda is only a prototype, NOT glmnet penalty equivalence.
    clf=LogisticRegression(
        penalty="elasticnet",solver="saga",l1_ratio=alpha,
        C=1.0/lam,max_iter=max_iter,tol=1e-5,
        random_state=622,fit_intercept=True)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        clf.fit(x,y)
    return clf.predict_proba(v)[:,1]

def run_elasticnet_inner(y_train,y_validation,train_x,validation_x,
                         alpha=0.5,lam=0.1):
    x=np.asarray(train_x,dtype=float)
    v=np.asarray(validation_x,dtype=float)
    if x.ndim!=2 or v.ndim!=2 or x.shape[1]!=v.shape[1]:
        raise ValueError("incompatible design matrices")
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite design matrix")
    if alpha not in (0.0,0.5,1.0) or not np.isfinite(lam) or lam<=0:
        raise ValueError("invalid alpha/lambda")
    return run_with_retry(
        y_train,y_validation,x,v,
        fit_primary=lambda a,b,c:_predict(a,b,c,alpha,lam,1000),
        fit_retry=lambda a,b,c:_predict(a,b,c,alpha,lam,2000))

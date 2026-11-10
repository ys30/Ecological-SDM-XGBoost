"""Preregistered one-retry fitting adapter with explicit failure provenance.

No model implementation is selected here. The caller must supply two
callables: primary fit and retry with doubled iterations/default optimizer.
"""
from dataclasses import dataclass
import numpy as np
from .class_support_policy import fallback_required,fallback_predict,relative_log_loss_skill,auc_eligible

@dataclass(frozen=True)
class FitOutcome:
    predictions: np.ndarray
    fallback: bool
    status: str
    attempt_count: int
    first_error: str
    retry_error: str
    n_training_positive: int
    n_validation_positive: int
    relative_log_loss_skill: float
    auc_eligible: bool

def _check_predictions(values,n):
    p=np.asarray(values,dtype=float)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("invalid predicted probabilities")
    return p

def run_with_retry(y_train,y_validation,train_x,validation_x,
                   fit_primary,fit_retry,minimum_positives=5, fitting_support_labels=None):
    tr=np.asarray(y_train)
    val=np.asarray(y_validation)
    if tr.ndim!=1 or val.ndim!=1 or not len(tr) or not len(val):
        raise ValueError("nonempty train and validation required")
    if not np.isin(tr,[0,1]).all() or not np.isin(val,[0,1]).all():
        raise ValueError("binary labels required")
    if len(train_x)!=len(tr) or len(validation_x)!=len(val):
        raise ValueError("feature/label row mismatch")
    if minimum_positives!=5:
        raise ValueError("preregistered positive threshold is five")
    support=tr if fitting_support_labels is None else np.asarray(fitting_support_labels)
    if support.ndim!=1 or not len(support) or not np.isin(support,[0,1]).all():
        raise ValueError("invalid fitting support labels")
    first_error=""
    retry_error=""
    attempts=0
    if fallback_required(support,minimum_positives) or np.unique(support).size<2:
        p=fallback_predict(tr,len(val))
        status="insufficient_positives"
        fallback=True
    else:
        attempts=1
        try:
            p=_check_predictions(fit_primary(train_x,tr,validation_x),len(val))
            status="fit_success"
            fallback=False
        except Exception as exc:
            first_error=f"{type(exc).__name__}: {exc}"
            attempts=2
            try:
                p=_check_predictions(fit_retry(train_x,tr,validation_x),len(val))
                status="retry_success"
                fallback=False
            except Exception as exc2:
                retry_error=f"{type(exc2).__name__}: {exc2}"
                p=fallback_predict(tr,len(val))
                status="fit_failure_fallback"
                fallback=True
    return FitOutcome(p,fallback,status,attempts,first_error,retry_error,
                      int(tr.sum()),int(val.sum()),
                      relative_log_loss_skill(tr,val,p),
                      auc_eligible(val,model_a_fallback=fallback))

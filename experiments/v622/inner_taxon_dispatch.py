"""Training-only class-support dispatch for one taxon and one inner split.

The caller supplies a fitting/prediction callback; this module does not choose a
model or access outer-test labels. Failed fitting is NOT silently swallowed.
"""
from dataclasses import dataclass
import numpy as np
from .class_support_policy import fallback_required,fallback_predict,relative_log_loss_skill,auc_eligible

@dataclass(frozen=True)
class InnerResult:
    predictions: np.ndarray
    fallback: bool
    reason: str
    n_training: int
    n_training_positive: int
    n_validation: int
    n_validation_positive: int
    relative_log_loss_skill: float
    validation_auc_defined: bool

def run_inner_taxon(y_train,y_validation,train_x,validation_x,fit_predict,
                    minimum_positives=5):
    tr=np.asarray(y_train)
    val=np.asarray(y_validation)
    if tr.ndim!=1 or val.ndim!=1 or not len(tr) or not len(val):
        raise ValueError("nonempty 1d train and validation labels required")
    if not np.isin(tr,[0,1]).all() or not np.isin(val,[0,1]).all():
        raise ValueError("binary labels required")
    if len(train_x)!=len(tr) or len(validation_x)!=len(val):
        raise ValueError("feature/label row count mismatch")
    if minimum_positives!=5:
        raise ValueError("v6.22 preregistered threshold is exactly five")
    is_fallback=fallback_required(tr,minimum_positives)
    if is_fallback:
        p=fallback_predict(tr,len(val))
        reason="training_positives_below_5"
    else:
        p=np.asarray(fit_predict(train_x,tr,validation_x),dtype=float)
        reason="fitted"
    if p.shape!=(len(val),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("invalid prediction shape or probabilities")
    skill=relative_log_loss_skill(tr,val,p)
    return InnerResult(p,is_fallback,reason,len(tr),int(tr.sum()),
                       len(val),int(val.sum()),skill,
                       auc_eligible(val,model_a_fallback=is_fallback))

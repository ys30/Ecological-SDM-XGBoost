"""EMS v6.22 preregistered class-support fallback and relative log-loss skill.

No model fitting. All prevalence values must be estimated on training rows.
"""
import numpy as np

EPS=1e-6

def training_prevalence(y):
    a=np.asarray(y,dtype=float)
    if a.ndim!=1 or a.size==0 or not np.isin(a,[0,1]).all():
        raise ValueError("binary, nonempty training labels required")
    return float(a.mean())

def fallback_required(y, minimum_positives=5):
    a=np.asarray(y,dtype=float)
    training_prevalence(a)
    return bool(a.sum()<minimum_positives)

def fallback_predict(y_train, n_validation):
    if n_validation<0:
        raise ValueError("invalid validation size")
    return np.full(n_validation,training_prevalence(y_train),dtype=float)

def null_prevalence(y_train):
    a=np.asarray(y_train,dtype=float)
    training_prevalence(a)
    return float((a.sum()+0.5)/(len(a)+1))

def log_loss(y_true, probabilities, epsilon=EPS):
    y=np.asarray(y_true,dtype=float)
    p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or y.size!=p.size or y.size==0:
        raise ValueError("nonempty, equal-length 1d arrays required")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("invalid labels or probabilities")
    p=np.clip(p,epsilon,1-epsilon)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def relative_log_loss_skill(y_train,y_validation,predictions):
    baseline=np.full(len(y_validation),null_prevalence(y_train))
    null_ll=log_loss(y_validation,baseline)
    return float(1-log_loss(y_validation,predictions)/null_ll)

def auc_eligible(y_validation, model_a_fallback=False, model_b_fallback=False):
    y=np.asarray(y_validation)
    if y.ndim!=1 or y.size==0 or not np.isin(y,[0,1]).all():
        raise ValueError("invalid validation labels")
    return bool(np.unique(y).size==2 and not model_a_fallback and not model_b_fallback)

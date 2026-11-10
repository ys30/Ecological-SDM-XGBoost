"""Engineering-only lambda path and inner-fold alpha/lambda selection.

Path follows glmnet-like lambda_max for alpha>0; ridge alpha=0 uses
an explicit effective alpha=0.001 convention. NOT verified glmnet default.
No outer predictions or scientific performance outputs.
"""
import numpy as np
from .elasticnet_prototype import run_elasticnet_inner
from .class_support_policy import fallback_required, fallback_predict, relative_log_loss_skill

ALPHAS=(0.,.5,1.)

def lambda_path(x,y,alpha,n_lambdas=30,ratio=1e-3):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or not len(y):
        raise ValueError("invalid train design")
    if not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise ValueError("invalid values")
    if alpha not in ALPHAS or n_lambdas!=30 or not 0<ratio<1:
        raise ValueError("invalid preregistered path")
    if len(np.unique(y))!=2:
        raise ValueError("lambda path requires two training classes")
    # Mean-centered gradient at the unpenalized intercept-only solution.
    gradient=np.abs((x.T@(y-y.mean()))/len(y))
    effective_alpha=max(alpha,0.001)
    maximum=max(float(gradient.max())/effective_alpha,1e-8)
    return np.geomspace(maximum,maximum*ratio,num=n_lambdas)

def select_alpha_lambda(inner_splits,alphas=ALPHAS):
    """Each split: (x_train,y_train,x_val,y_val); no outer rows permitted by caller."""
    if tuple(alphas)!=ALPHAS:
        raise ValueError("preregistered alpha grid required")
    splits=list(inner_splits)
    if not splits:
        raise ValueError("at least one inner split required")
    # Candidate paths are constructed independently on every inner-training
    # partition. The same path index (0..29), rather than a lambda value
    # borrowed from a different fold, is evaluated across folds.
    # This is an explicit engineering convention pending R glmnet path audit.
    paths={}
    for alpha in alphas:
        paths[alpha]=[lambda_path(x,y,alpha) for x,y,_,_ in splits]
    records=[]
    for alpha in alphas:
        for index in range(30):
            skills=[]
            for (x,y,v,z),path in zip(splits,paths[alpha]):
                lam=float(path[index])
                result=run_elasticnet_inner(y,z,x,v,alpha=alpha,lam=lam)
                skills.append(result.relative_log_loss_skill)
            records.append((float(np.mean(skills)),alpha,index))
    # Deterministic tie break: alpha then path index.
    best=sorted(records,key=lambda row:(-row[0],row[1],row[2]))[0]
    return {"alpha":best[1],"lambda_index":best[2],
            "mean_skill":best[0],"n_candidates":len(records),
            "refit_lambda_policy":"recompute selected path index on full outer training"}

def resolve_refit_lambda(outer_train_x,outer_train_y,selection):
    """Resolve selected fold-local path index on outer-training data only.

    Returns hyperparameters, not a fitted model or outer predictions.
    """
    if not isinstance(selection,dict):
        raise ValueError("selection must be a dictionary")
    alpha=selection.get("alpha")
    index=selection.get("lambda_index")
    if alpha not in ALPHAS or type(index) is not int or not 0<=index<30:
        raise ValueError("invalid selected alpha or path index")
    path=lambda_path(outer_train_x,outer_train_y,alpha)
    return {"alpha":alpha,"lambda":float(path[index]),
            "lambda_index":index,"n_outer_training":len(outer_train_y)}

def equal_taxon_mean_skill(taxon_fold_scores):
    """Equal taxon weighting, then equal evaluable folds within each taxon.

    Input: iterable of (taxon_id, fold_id, relative_log_loss_skill).
    Missing/unevaluable folds are excluded; taxon coverage is reported.
    """
    from collections import defaultdict
    scores=defaultdict(dict)
    for taxon,fold,skill in taxon_fold_scores:
        if not np.isfinite(skill):
            raise ValueError("nonfinite skill")
        if fold in scores[taxon]:
            raise ValueError("duplicate taxon/fold")
        scores[taxon][fold]=float(skill)
    if not scores:
        raise ValueError("no evaluable taxon-fold scores")
    means={taxon:float(np.mean(list(folds.values())))
           for taxon,folds in scores.items()}
    return {"mean_skill":float(np.mean(list(means.values()))),
            "n_taxa":len(means),
            "n_taxon_folds":sum(len(v) for v in scores.values()),
            "per_taxon_mean":means}

def select_alpha_lambda_by_taxon(taxon_splits, alphas=ALPHAS):
    """Engineering-only joint search with equal taxon weight.

    taxon_splits: mapping taxon -> iterable of (fold_id, x_train, y_train,
    x_validation, y_validation). Every split must already exclude outer rows.
    Paths are local to each taxon/fold. Returns index, not a globally shared
    lambda. Not R glmnet/cv.glmnet-equivalent.
    """
    if tuple(alphas)!=ALPHAS:
        raise ValueError("preregistered alpha grid required")
    splits={}
    for taxon,items in taxon_splits.items():
        entries=list(items)
        if not entries:
            raise ValueError("empty taxon")
        seen=set()
        for fold,x,y,v,z in entries:
            if fold in seen:
                raise ValueError("duplicate taxon/fold")
            seen.add(fold)
            if len(y)==0 or len(z)==0:
                raise ValueError("empty inner partition")
        splits[taxon]=entries
    if not splits:
        raise ValueError("no taxa")
    records=[]
    for alpha in alphas:
        paths={}
        for taxon,entries in splits.items():
            for fold,x,y,v,z in entries:
                paths[(taxon,fold)]=(None if fallback_required(y)
                                     else lambda_path(x,y,alpha))
        for index in range(30):
            scored=[]
            fallback_count=0
            status_counts={}
            for taxon,entries in splits.items():
                for fold,x,y,v,z in entries:
                    path=paths[(taxon,fold)]
                    if path is None:
                        # No optimizer or lambda path for <5 training positives.
                        predictions=fallback_predict(y,len(z))
                        skill=relative_log_loss_skill(y,z,predictions)
                        status="insufficient_positives"
                    else:
                        lam=float(path[index])
                        result=run_elasticnet_inner(
                            y,z,x,v,alpha=alpha,lam=lam)
                        skill=result.relative_log_loss_skill
                        status=result.status
                    fallback_count+=int(status in (
                        "insufficient_positives","fit_failure_fallback"))
                    status_counts[status]=status_counts.get(status,0)+1
                    scored.append((taxon,fold,skill))
            summary=equal_taxon_mean_skill(scored)
            records.append((summary["mean_skill"],alpha,index,summary,
                            fallback_count,status_counts))
    best=sorted(records,key=lambda row:(-row[0],row[1],row[2]))[0]
    return {"alpha":best[1],"lambda_index":best[2],
            "mean_skill":best[0],"n_candidates":len(records),
            "n_taxa":best[3]["n_taxa"],
            "n_taxon_folds":best[3]["n_taxon_folds"],
            "n_fallback_taxon_folds":best[4],
            "fit_status_counts":best[5],
            "refit_lambda_policy":"recompute per taxon on full outer training"}

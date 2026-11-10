"""Engineering-only lambda path and inner-fold alpha/lambda selection.

Path follows glmnet-like lambda_max for alpha>0; ridge alpha=0 uses
an explicit effective alpha=0.001 convention. NOT verified glmnet default.
No outer predictions or scientific performance outputs.
"""
import numpy as np
from .elasticnet_prototype import run_elasticnet_inner

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

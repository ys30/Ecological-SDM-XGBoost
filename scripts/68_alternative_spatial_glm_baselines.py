"""Fold-local per-taxon logistic GLM baselines on alternative geographic folds.

Two specifications: linear four environmental predictors; quadratic adds squared
rainfall and temperature. L2 regularization guards sparse-taxon separation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
d=pd.read_csv("data/train.csv")
folds=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
assert folds["plot"].is_unique and folds.fold.nunique()==5
d=d.merge(folds[["plot","fold"]],on="plot",validate="many_to_one")
assert len(d)==9336 and d.fold.notna().all() and d.id.is_unique
features=["rainann","tempann","soildepth","topo"]
assert d[features].notna().all().all()
out=Path("results/alternative_spatial_validation/glm_baselines");out.mkdir(parents=True,exist_ok=True)
predictions=[];diagnostics=[]
for fold in sorted(d.fold.unique()):
    tr=d[d.fold!=fold].copy();te=d[d.fold==fold].copy()
    assert set(tr["plot"]).isdisjoint(te["plot"])
    scaler=StandardScaler().fit(tr[features])
    ztr=scaler.transform(tr[features]);zte=scaler.transform(te[features])
    for variant in ["linear","quadratic"]:
        xtr=ztr if variant=="linear" else np.column_stack([ztr,ztr[:,0]**2,ztr[:,1]**2])
        xte=zte if variant=="linear" else np.column_stack([zte,zte[:,0]**2,zte[:,1]**2])
        for taxon in sorted(d.Species.unique()):
            ti=np.flatnonzero(tr.Species.to_numpy()==taxon)
            vi=np.flatnonzero(te.Species.to_numpy()==taxon)
            assert len(ti)>0 and len(vi)>0
            y=tr["pres.abs"].to_numpy()[ti].astype(int)
            if len(np.unique(y))<2:
                # Train-fold-only Laplace-smoothed intercept-only fallback;
                # never fit an impossible one-class logistic regression.
                p=np.full(len(vi),(y.sum()+1)/(len(y)+2),dtype=float)
                status="one_class_intercept_only"
            else:
                model=LogisticRegression(penalty="l2",C=100.0,solver="lbfgs",
                    max_iter=3000,tol=1e-8,random_state=20261104)
                model.fit(xtr[ti],y)
                p=model.predict_proba(xte[vi])[:,1]
                status="l2_logistic"
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            g=te.iloc[vi]
            predictions.append(pd.DataFrame(dict(id=g.id.to_numpy(),plot=g["plot"].to_numpy(),
                species=taxon,fold=fold,observed=g["pres.abs"].to_numpy(),
                probability=p,variant="glm_"+variant)))
            diagnostics.append(dict(fold=fold,species=taxon,variant=variant,
                n_train=len(ti),positives_train=int(y.sum()),status=status))
pred=pd.concat(predictions,ignore_index=True)
assert len(pred)==18672 and pred.groupby("variant").size().eq(9336).all()
for variant,g in pred.groupby("variant"):
    assert g.id.is_unique
    assert set(g.id)==set(d.id)
pred.to_csv(out/"glm_linear_quadratic_oof.csv",index=False)
pd.DataFrame(diagnostics).to_csv(out/"fit_diagnostics.csv",index=False)
def metrics(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    return pd.Series(dict(n=len(g),positives=int(y.sum()),
        auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        ap=average_precision_score(y,p) if y.sum()>0 else np.nan,
        brier=np.mean((y-p)**2),logloss=log_loss(y,p,labels=[0,1])))
m=pred.groupby(["variant","species","fold"]).apply(metrics,include_groups=False).reset_index()
m.to_csv(out/"taxon_fold_metrics.csv",index=False)
pooled=pred.groupby("variant").apply(metrics,include_groups=False).reset_index()
pooled.to_csv(out/"pooled_metrics.csv",index=False)
print("POOLED\n",pooled.to_string(index=False))
print("FIT STATUS\n",pd.DataFrame(diagnostics).groupby(["variant","status"]).size().to_string())
with open(out/"GLM_BASELINES_VERIFIED.md","w") as f:
    f.write("# Alternative-partition per-taxon logistic regression baselines\n\n")
    f.write("Models are fitted independently by anonymous taxon and fold. The linear model uses rainfall, temperature, soil depth and topography; quadratic adds squared fold-standardized rainfall and temperature. All standardization uses training folds only. These are **L2-regularized logistic GLMs** (C=100), not unpenalized maximum-likelihood GLMs. One-class training taxa use a smoothed intercept-only fallback. Both models use identical test records and the second within-dataset geographic partition; no external validation.\n\n")
    f.write("## Pooled OOF metrics\n\n"+pooled.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("## Fitting diagnostics\n\n"+pd.DataFrame(diagnostics).groupby(["variant","status"]).size().to_string()+"\n\n")
    f.write("Interpretation must distinguish per-taxon logistic models from hierarchical Bayesian partial pooling. Predictive comparisons against Bayesian models should use identical OOF IDs and species-fold evaluation denominators.\n")

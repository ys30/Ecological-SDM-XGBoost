"""Leakage-safe buffered refit of per-taxon GLMs on alternative spatial folds.

For each fold, exclude training plots within radius of ANY held-out plot,
then fit on remaining training plots and predict every held-out plot.
Distances are in projected easting/northing units (assumed meters).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
d=d.merge(a[["plot","fold"]],on="plot",validate="many_to_one")
assert len(d)==9336 and d.id.is_unique
p=d[["plot","easting","northing","fold"]].drop_duplicates("plot")
assert len(p)==1167 and p[["easting","northing"]].notna().all().all()
feat=["rainann","tempann","soildepth","topo"]
rows=[];diag=[]
for km in [0,5,10,20]:
 for fold in range(1,6):
    testplots=p[p.fold==fold];trainplots=p[p.fold!=fold]
    distances=cKDTree(testplots[["easting","northing"]].to_numpy()).query(
        trainplots[["easting","northing"]].to_numpy(),k=1)[0]
    kept=set(trainplots.loc[distances>=km*1000,"plot"])
    tr=d[(d.fold!=fold)&d["plot"].isin(kept)].copy()
    te=d[d.fold==fold].copy()
    assert set(tr["plot"]).isdisjoint(te["plot"])
    assert len(te)>0 and len(tr)>0
    scaler=StandardScaler().fit(tr[feat])
    ztr=scaler.transform(tr[feat]);zte=scaler.transform(te[feat])
    for variant in ["linear","quadratic"]:
      xx=ztr if variant=="linear" else np.column_stack([ztr,ztr[:,0]**2,ztr[:,1]**2])
      xt=zte if variant=="linear" else np.column_stack([zte,zte[:,0]**2,zte[:,1]**2])
      for taxon in sorted(d.Species.unique()):
        ti=np.flatnonzero(tr.Species.to_numpy()==taxon)
        vi=np.flatnonzero(te.Species.to_numpy()==taxon)
        assert len(ti)>0 and len(vi)>0
        y=tr["pres.abs"].to_numpy()[ti].astype(int)
        if len(np.unique(y))<2:
            prob=np.full(len(vi),(y.sum()+1)/(len(y)+2))
            status="one_class_smoothed_intercept"
        else:
            model=LogisticRegression(C=100,solver="lbfgs",max_iter=3000,tol=1e-8,random_state=20261104)
            model.fit(xx[ti],y)
            prob=model.predict_proba(xt[vi])[:,1]
            status="l2_logistic"
        g=te.iloc[vi]
        rows.append(pd.DataFrame(dict(id=g.id.to_numpy(),species=taxon,fold=fold,
            observed=g["pres.abs"].to_numpy(),probability=prob,
            variant="glm_"+variant,buffer_km=km)))
        diag.append(dict(buffer_km=km,fold=fold,species=taxon,variant=variant,
            n_train_plots=len(kept),n_excluded_plots=len(trainplots)-len(kept),
            positives_train=int(y.sum()),status=status))
pred=pd.concat(rows,ignore_index=True)
assert pred.groupby(["buffer_km","variant"]).size().eq(9336).all()
def score(g):
    y=g.observed.to_numpy();pr=g.probability.to_numpy()
    return pd.Series(dict(n=len(y),positives=int(y.sum()),
        auc=roc_auc_score(y,pr) if y.min()!=y.max() else np.nan,
        ap=average_precision_score(y,pr) if y.sum()>0 else np.nan,
        brier=np.mean((y-pr)**2),logloss=log_loss(y,pr,labels=[0,1])))
pooled=pred.groupby(["buffer_km","variant"]).apply(score,include_groups=False).reset_index()
cells=pred.groupby(["buffer_km","variant","species","fold"]).apply(score,include_groups=False).reset_index()
out=Path("results/alternative_spatial_validation/buffered_glm")
out.mkdir(parents=True,exist_ok=True)
pred.to_csv(out/"buffered_glm_oof.csv",index=False)
pooled.to_csv(out/"pooled_metrics.csv",index=False)
cells.to_csv(out/"taxon_fold_metrics.csv",index=False)
pd.DataFrame(diag).to_csv(out/"training_exclusion_diagnostics.csv",index=False)
print("POOLED\n"+pooled.to_string(index=False))
print("EXCLUSION\n"+pd.DataFrame(diag).groupby(["buffer_km","fold"])[["n_train_plots","n_excluded_plots"]].first().to_string())
with open(out/"BUFFERED_GLM_AUDIT.md","w") as f:
 f.write("# Buffered alternative-spatial GLM refit\n\n")
 f.write("For each held-out fold, all training plots less than the specified projected Euclidean distance from **any held-out plot** are removed **before** standardization and refitting. Evaluation keeps the same held-out records at all radii. 0/5/10/20 km are buffer radii, not measured nearest-neighbor test distances after exclusion. Distances assume projected easting/northing are meters; CRS metadata should be verified before physical interpretation. This experiment covers **per-taxon regularized GLMs only**, not Bayesian or XGBoost, and is within-dataset spatial sensitivity, not external validation.\n\n")
 f.write(pooled.to_markdown(index=False,floatfmt=".6f")+"\n")

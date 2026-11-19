#!/usr/bin/env python3
"""Primary-fold 10km matched-buffer GLM refits. Independent of frozen v6.14."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,average_precision_score
from importlib.machinery import SourceFileLoader
xgb=SourceFileLoader("v623_xgb","scripts/102_v623_fixed_xgb_outer_oof.py").load_module()
OUT=Path("results/v623_matched_glm_10km")
FEATURES=xgb.FEATURES
def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    data=xgb.read("data/train.csv")
    folds=xgb.read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    assert len(fmap)==len(folds)
    coords={}
    for r in data:
        xy=(float(r["long"]),float(r["lat"]))
        if r["plot"] in coords:assert coords[r["plot"]]==xy
        coords[r["plot"]]=xy
    assert set(coords)==set(fmap)
    taxa=sorted({r["Species"] for r in data})
    pred=[];metrics=[];diag=[]
    for fold in range(1,6):
        testplots=sorted((p for p in fmap if fmap[p]==fold),key=int)
        pool=sorted((p for p in fmap if fmap[p]!=fold),key=int)
        distances=xgb.km(np.asarray([coords[p] for p in pool]),
                         np.asarray([coords[p] for p in testplots])).min(axis=1)
        keep={p for p,d in zip(pool,distances) if d>10.0}
        for taxon in taxa:
            tr=[r for r in data if r["Species"]==taxon and r["plot"] in keep]
            te=[r for r in data if r["Species"]==taxon and r["plot"] in testplots]
            xtr=np.asarray([[float(r[k]) for k in FEATURES] for r in tr])
            xte=np.asarray([[float(r[k]) for k in FEATURES] for r in te])
            ytr=np.asarray([int(r["pres.abs"]) for r in tr])
            yte=np.asarray([int(r["pres.abs"]) for r in te])
            if len(np.unique(ytr))!=2:raise ValueError(f"One class: {fold} {taxon}")
            scaler=StandardScaler().fit(xtr)
            ztr=scaler.transform(xtr);zte=scaler.transform(xte)
            for variant in ("linear","quadratic"):
                a=ztr if variant=="linear" else np.column_stack([ztr,ztr[:,0]**2,ztr[:,2]**2])
                b=zte if variant=="linear" else np.column_stack([zte,zte[:,0]**2,zte[:,2]**2])
                model=LogisticRegression(C=100,solver="lbfgs",max_iter=3000,tol=1e-8,random_state=20261104)
                model.fit(a,ytr)
                pp=np.clip(model.predict_proba(b)[:,1],1e-12,1-1e-12)
                for r,p in zip(te,pp):
                    pred.append(dict(id=r["id"],plot=r["plot"],taxon=taxon,outer_fold=fold,
                                     y=int(r["pres.abs"]),p=float(p),variant=variant))
                eligible=len(np.unique(yte))==2
                metrics.append(dict(variant=variant,taxon=taxon,outer_fold=fold,
                    n_test=len(yte),positives_test=int(yte.sum()),n_train=len(ytr),
                    positives_train=int(ytr.sum()),logloss=float(log_loss(yte,pp,labels=[0,1])),
                    brier=float(brier_score_loss(yte,pp)),
                    auc=float(roc_auc_score(yte,pp)) if eligible else "",
                    ap=float(average_precision_score(yte,pp)) if eligible else "",
                    auc_ap_eligible=eligible))
                diag.append(dict(variant=variant,taxon=taxon,outer_fold=fold,
                    n_excluded_train_plots=len(pool)-len(keep),n_train_plots=len(keep),
                    train_id_sha256=xgb.ids_sha256(r["id"] for r in tr),
                    test_id_sha256=xgb.ids_sha256(r["id"] for r in te)))
    assert len(pred)==2*len(data)
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/"glm_oof_predictions.csv",pred)
    write(OUT/"glm_fold_metrics.csv",metrics)
    write(OUT/"glm_buffer_diagnostics.csv",diag)
    manifest=dict(status="MATCHED_PRIMARY_FOLD_10KM_GLM",variants=["linear","quadratic"],
        rows_per_variant=len(data),features=FEATURES,buffer_rule="same haversine >10 km as v6.23 XGB",
        fold_sha256=xgb.sha("results/spatial_2d_block_assignments.csv"),
        data_sha256=xgb.sha("data/train.csv"),
        scaling="fold-local train-only StandardScaler",C=100,solver="lbfgs",
        quadratic_terms=["standardized_rainann_squared","standardized_tempann_squared"],
        caveat="Quadratic feature set prespecified from historical GLM implementation; not optimized on v6.23 held-out outcomes.")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(fits=len(metrics),oof_rows=len(pred),per_variant=len(data)),indent=2))
if __name__=="__main__":main()

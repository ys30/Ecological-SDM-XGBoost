#!/usr/bin/env python3
"""v6.23 predeclared fixed XGBoost baseline: 5 outer folds x 8 taxa, 10-km test buffer.

No hyperparameter search. Outputs are distinct from frozen historical results.
"""
import csv,hashlib,json,os,platform
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,average_precision_score
from xgboost import XGBClassifier
import xgboost,sklearn

SEED=622
BUFFER_KM=10.0
FEATURES=["rainann","soildepth","tempann","topo"]
PARAMS=dict(max_depth=2,learning_rate=0.03,n_estimators=100,min_child_weight=1,
            subsample=0.8,colsample_bytree=0.8)
OUT=Path("results/v623_fixed_xgb_outer_oof")

def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def km(a,b):
    lo1,la1=np.radians(a[:,0])[:,None],np.radians(a[:,1])[:,None]
    lo2,la2=np.radians(b[:,0])[None,:],np.radians(b[:,1])[None,:]
    q=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(q,0,1)))
def write(path,records):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def main():
    source="data/train.csv";foldpath="results/spatial_2d_block_assignments.csv"
    rows=read(source);foldrows=read(foldpath)
    fmap={r["plot"]:int(r["fold"]) for r in foldrows}
    if len(fmap)!=len(foldrows):raise ValueError("Duplicate outer fold")
    taxa=sorted({r["Species"] for r in rows})
    if len(taxa)!=8 or set(fmap.values())!=set(range(1,6)):raise ValueError("Unexpected folds/taxa")
    coords={}
    for r in rows:
        p=r["plot"];xy=(float(r["long"]),float(r["lat"]))
        if p in coords and coords[p]!=xy:raise ValueError("Inconsistent plot coordinates")
        coords[p]=xy
    if set(coords)!=set(fmap) or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Canonical data mismatch")
    OUT.mkdir(parents=True,exist_ok=True)
    preds=[];metrics=[];diagnostics=[]
    for outer in range(1,6):
        testplots={p for p in fmap if fmap[p]==outer}
        pool=sorted((p for p in fmap if fmap[p]!=outer),key=int)
        testxy=np.asarray([coords[p] for p in sorted(testplots,key=int)])
        distances=km(np.asarray([coords[p] for p in pool]),testxy).min(axis=1)
        keep={p for p,d in zip(pool,distances) if d>BUFFER_KM}
        if any(d<=BUFFER_KM for p,d in zip(pool,distances) if p in keep):
            raise AssertionError("Buffer leakage")
        for taxon in taxa:
            train=[r for r in rows if r["Species"]==taxon and r["plot"] in keep]
            test=[r for r in rows if r["Species"]==taxon and r["plot"] in testplots]
            if not train or not test:raise ValueError("Empty outer split")
            xtr=np.asarray([[float(r[k]) for k in FEATURES] for r in train])
            xte=np.asarray([[float(r[k]) for k in FEATURES] for r in test])
            ytr=np.asarray([int(r["pres.abs"]) for r in train])
            yte=np.asarray([int(r["pres.abs"]) for r in test])
            if len(np.unique(ytr))!=2 or not np.isfinite(xtr).all() or not np.isfinite(xte).all():
                raise ValueError(f"Invalid training data outer={outer} taxon={taxon}")
            model=XGBClassifier(objective="binary:logistic",eval_metric="logloss",
                random_state=SEED,n_jobs=2,tree_method="hist",**PARAMS)
            model.fit(xtr,ytr)
            p=np.clip(model.predict_proba(xte)[:,1].astype(float),1e-12,1-1e-12)
            for r,prob in zip(test,p):
                preds.append(dict(id=r["id"],plot=r["plot"],taxon=taxon,outer_fold=outer,
                                  y=int(r["pres.abs"]),p=float(prob)))
            eligible=len(np.unique(yte))==2
            base=float(ytr.mean())
            baseline=np.repeat(np.clip(base,1e-12,1-1e-12),len(yte))
            ll=float(log_loss(yte,p,labels=[0,1]))
            null_ll=float(log_loss(yte,baseline,labels=[0,1]))
            metrics.append(dict(outer_fold=outer,taxon=taxon,n_test=len(yte),
                positives_test=int(yte.sum()),n_train=len(ytr),positives_train=int(ytr.sum()),
                logloss=ll,baseline_logloss=null_ll,relative_logloss=1-ll/null_ll if null_ll>1e-10 else "",
                brier=float(brier_score_loss(yte,p)),
                auc=float(roc_auc_score(yte,p)) if eligible else "",
                ap=float(average_precision_score(yte,p)) if eligible else "",
                auc_ap_eligible=eligible))
            diagnostics.append(dict(outer_fold=outer,taxon=taxon,train_n=len(ytr),
                train_positive=int(ytr.sum()),test_n=len(yte),test_positive=int(yte.sum()),
                buffered_out_train_plots=len(pool)-len(keep)))
    if len(preds)!=len(rows) or len({r["id"] for r in preds})!=len(rows):
        raise AssertionError("OOF coverage incomplete")
    write(OUT/"outer_oof_predictions.csv",preds)
    write(OUT/"outer_fold_metrics.csv",metrics)
    write(OUT/"outer_fold_diagnostics.csv",diagnostics)
    manifest=dict(status="FIXED_CONFIG_OUTER_10KM_BUFFER_BASELINE",fit_count=len(metrics),
        oof_rows=len(preds),taxa=taxa,features=FEATURES,parameters=PARAMS,seed=SEED,
        outer_buffer_km=BUFFER_KM,buffer_rule="exclude training plots <=10 km from any outer test plot",
        data_sha256=sha(source),fold_sha256=sha(foldpath),python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,xgboost_version=xgboost.__version__,
        git_sha=os.environ.get("GITHUB_SHA",""),
        limitations=["Fixed configuration, NOT nested tuned","Inner spatial validation not used",
            "10-km outer buffer is not proof of protocol equivalence to historical comparator",
            "AUC/AP undefined for single-class outer tests"])
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(fits=len(metrics),oof_rows=len(preds),
        auc_ap_ineligible=sum(not r["auc_ap_eligible"] for r in metrics)),indent=2))
if __name__=="__main__":main()

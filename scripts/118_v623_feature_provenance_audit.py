#!/usr/bin/env python3
"""Audit observable feature integrity and fold-local transformations; not source provenance."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from importlib.machinery import SourceFileLoader
xgb=SourceFileLoader("v623_xgb_features","scripts/102_v623_fixed_xgb_outer_oof.py").load_module()
OUT=Path("results/v623_matched_three_model")
def read(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
    rows=read("data/train.csv");folds=read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    assert len(fmap)==len(folds)
    features=xgb.FEATURES
    assert features==["rainann","soildepth","tempann","topo"]
    coords={}
    for r in rows:
        p=r["plot"];xy=(float(r["long"]),float(r["lat"]))
        if p in coords and coords[p]!=xy:raise ValueError("Inconsistent coordinates")
        coords[p]=xy
    assert set(coords)==set(fmap)
    taxa=sorted({r["Species"] for r in rows})
    assert len(taxa)==8
    records=[]
    for outer in range(1,6):
        testplots=sorted((p for p in fmap if fmap[p]==outer),key=int)
        pool=sorted((p for p in fmap if fmap[p]!=outer),key=int)
        distance=xgb.km(np.asarray([coords[p] for p in pool]),np.asarray([coords[p] for p in testplots])).min(axis=1)
        keep={p for p,d in zip(pool,distance) if d>10}
        for taxon in taxa:
            train=[r for r in rows if r["Species"]==taxon and r["plot"] in keep]
            test=[r for r in rows if r["Species"]==taxon and r["plot"] in testplots]
            xtr=np.array([[float(r[f]) for f in features] for r in train])
            xte=np.array([[float(r[f]) for f in features] for r in test])
            assert np.isfinite(xtr).all() and np.isfinite(xte).all()
            scaler=StandardScaler().fit(xtr)
            ztr=scaler.transform(xtr);zte=scaler.transform(xte)
            assert np.allclose(scaler.mean_,xtr.mean(axis=0))
            assert np.isfinite(ztr).all() and np.isfinite(zte).all()
            for i,f in enumerate(features):
                records.append(dict(outer_fold=outer,taxon=taxon,feature=f,
                    train_n=len(train),test_n=len(test),train_min=float(xtr[:,i].min()),
                    train_max=float(xtr[:,i].max()),test_min=float(xte[:,i].min()),
                    test_max=float(xte[:,i].max()),train_mean=float(scaler.mean_[i]),
                    train_scale=float(scaler.scale_[i]),
                    test_outside_train_range=int(np.sum((xte[:,i]<xtr[:,i].min())|(xte[:,i]>xtr[:,i].max()))),
                    status="PASS"))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"feature_fold_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=dict(status="PASS_OBSERVABLE_FEATURE_CHECKS",taxa=len(taxa),outer_folds=5,
        features=features,cells=len(records),test_outside_train_range_total=sum(r["test_outside_train_range"] for r in records),
        verified=["four numeric finite predictors","training-only StandardScaler reconstructed for GLM","test observations excluded from reconstructed scaler fit","10-km outer exclusion reconstructed"],
        not_verified=["original source and acquisition of each predictor","whether spatial folds were designed without inspecting outcomes","whether predictor selection used test outcomes in prior research","whether input data include target leakage from upstream joins","independent geographic or temporal external validation"],
        interpretation="A code-and-data integrity check; NOT proof of absence of upstream leakage.")
    (OUT/"feature_provenance_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

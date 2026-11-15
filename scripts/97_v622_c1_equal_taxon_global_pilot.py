#!/usr/bin/env python3
"""v6.22 C1: one-outer-fold, nested spatial XGBoost pilot. No historical outputs overwritten."""
import csv, hashlib, json, math, os, platform, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import sklearn, xgboost

SEED=622; OUTER=1; INNER=4
FEATURES=["rainann","soildepth","tempann","topo"]
OUT=Path("results/v622_c1_global_pilot")
def read(path):
    with open(path,newline="",encoding="utf-8-sig") as fh: return list(csv.DictReader(fh))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score(y,p,base):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    ll=float(log_loss(y,p,labels=[0,1]))
    null=float(log_loss(y,np.repeat(np.clip(base,1e-12,1-1e-12),len(y)),labels=[0,1]))
    return ll, (1-ll/null if null>1e-10 else float("nan"))
def choose_global(scores):
    """Select a single candidate by equal-taxon mean of inner-fold mean scores."""
    x=np.asarray(scores,dtype=float)
    if x.shape!=(8,30) or not np.isfinite(x).all():
        raise ValueError("Expected finite 8 taxa x 30 candidates")
    return int(np.argmax(x.mean(axis=0)))

def candidates():
    # Exactly 30 deterministic candidates.
    return [dict(max_depth=depth,learning_rate=eta,n_estimators=rounds,
                 min_child_weight=1,subsample=.8,colsample_bytree=.8)
      for depth in [2,3,4] for eta in [.03,.10] for rounds in [100,250,400,600,800]]
def main():
    source="data/train.csv"; foldpath="results/spatial_2d_block_assignments.csv"
    rows=read(source); foldrows=read(foldpath)
    fmap={r["plot"]:int(r["fold"]) for r in foldrows}
    assert len(fmap)==len(foldrows)
    ids=[r["id"] for r in rows]; assert len(set(ids))==len(ids)
    taxa=sorted({r["Species"] for r in rows}); assert len(taxa)==8
    assert {r["plot"] for r in rows}==set(fmap)
    assert set(fmap.values())==set(range(1,6))
    plots=sorted({r["plot"] for r in rows if fmap[r["plot"]]!=OUTER},key=int)
    plotcoords={}
    for r in rows:
        if r["plot"] in plots: plotcoords[r["plot"]]=(float(r["long"]),float(r["lat"]))
    # Outcome-blind spatial grouping, computed from outer training plots only.
    coords=np.array([plotcoords[p] for p in plots])
    km=KMeans(n_clusters=INNER,random_state=SEED,n_init=10).fit(coords)
    inner={p:int(c) for p,c in zip(plots,km.labels_)}
    assert set(inner.values())==set(range(INNER))
    configs=candidates(); assert len(configs)==30
    OUT.mkdir(parents=True,exist_ok=True)
    diagnostics=[]
    for taxon in taxa:
        train=[r for r in rows if r["Species"]==taxon and fmap[r["plot"]]!=OUTER]
        for innerfold in range(INNER):
            tr=[r for r in train if inner[r["plot"]]!=innerfold]
            va=[r for r in train if inner[r["plot"]]==innerfold]
            diagnostics.append(dict(taxon=taxon,inner_fold=innerfold,
                train_n=len(tr),train_positive=sum(int(r["pres.abs"]) for r in tr),
                validation_n=len(va),validation_positive=sum(int(r["pres.abs"]) for r in va),
                train_classes=len({r["pres.abs"] for r in tr}),
                status="PASS" if len({r["pres.abs"] for r in tr})==2 else "FAIL_SINGLE_CLASS"))
    with (OUT/"c1_inner_fold_diagnostics.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(diagnostics[0]))
        writer.writeheader();writer.writerows(diagnostics)
    failed=[r for r in diagnostics if r["status"]!="PASS"]
    if failed:
        raise ValueError(str(len(failed))+" inner training splits single-class; see c1_inner_fold_diagnostics.csv")
    innerout=[]; outerout=[]; chosen=[]
    taxon_scores={}
    for taxon in taxa:
        tax=[r for r in rows if r["Species"]==taxon]
        train=[r for r in tax if fmap[r["plot"]]!=OUTER]
        test=[r for r in tax if fmap[r["plot"]]==OUTER]
        assert not ({r["plot"] for r in train}&{r["plot"] for r in test})
        assert len(train)>0 and len(test)>0
        def xy(rr):
            return np.array([[float(r[k]) for k in FEATURES] for r in rr]),np.array([int(r["pres.abs"]) for r in rr])
        xtrain,ytrain=xy(train); xout,yout=xy(test)
        assert np.isfinite(xtrain).all() and np.isfinite(xout).all()
        scores=[]
        for ci,config in enumerate(configs):
            foldvalues=[]
            for innerfold in range(INNER):
                itr=[r for r in train if inner[r["plot"]]!=innerfold]
                iva=[r for r in train if inner[r["plot"]]==innerfold]
                assert itr and iva
                xi,yi=xy(itr); xv,yv=xy(iva)
                assert not ({r["plot"] for r in itr}&{r["plot"] for r in iva})
                if len(np.unique(yi))<2:
                    raise ValueError(f"Single-class inner training taxon={taxon} fold={innerfold}")
                model=XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=SEED,
                    n_jobs=2,tree_method="hist",**config)
                model.fit(xi,yi)
                pred=model.predict_proba(xv)[:,1]
                ll,rel=score(yv,pred,float(yi.mean()))
                if not np.isfinite(rel): raise ValueError("Undefined relative log loss")
                foldvalues.append(rel)
                innerout.append(dict(taxon=taxon,candidate=ci,inner_fold=innerfold,
                    n_train=len(yi),n_validation=len(yv),positive_train=int(yi.sum()),
                    positive_validation=int(yv.sum()),logloss=ll,relative_logloss=rel))
            scores.append(float(np.mean(foldvalues)))
        taxon_scores[taxon]=scores
    # Persist all inner-fold scores before global selection/refits; artifacts
    # remain available if a later stage fails.
    with (OUT/"c1_inner_scores.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(innerout[0]))
        writer.writeheader()
        writer.writerows(innerout)
    with (OUT/"c1_candidate_summary.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=["candidate","taxon","mean_relative_logloss"])
        writer.writeheader()
        for taxon in taxa:
            for ci,value in enumerate(taxon_scores[taxon]):
                writer.writerow(dict(candidate=ci,taxon=taxon,mean_relative_logloss=value))
    # Global candidate selected once from equal-weighted taxon scores.
    matrix=np.array([taxon_scores[t] for t in taxa],dtype=float)
    assert matrix.shape==(8,30) and np.isfinite(matrix).all()
    global_scores=matrix.mean(axis=0)
    best=choose_global(matrix)
    chosen=[dict(scope="equal_taxon_global",candidate=best,
                 mean_relative_logloss=float(global_scores[best]),**configs[best])]
    for taxon in taxa:
        tax=[r for r in rows if r["Species"]==taxon]
        train=[r for r in tax if fmap[r["plot"]]!=OUTER]
        test=[r for r in tax if fmap[r["plot"]]==OUTER]
        xtrain=np.array([[float(r[k]) for k in FEATURES] for r in train])
        ytrain=np.array([int(r["pres.abs"]) for r in train])
        xout=np.array([[float(r[k]) for k in FEATURES] for r in test])
        if len(np.unique(ytrain))<2: raise ValueError("Single-class outer training")
        model=XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=SEED,
            n_jobs=2,tree_method="hist",**configs[best])
        model.fit(xtrain,ytrain)
        pp=model.predict_proba(xout)[:,1]
        for r,p in zip(test,pp):
            outerout.append(dict(id=r["id"],plot=r["plot"],taxon=taxon,outer_fold=OUTER,
                    y=int(r["pres.abs"]),p=float(p),candidate=best))
    assert len({r["id"] for r in outerout})==len(outerout)
    assert len(outerout)==sum(fmap[r["plot"]]==OUTER for r in rows)
    assert all(math.isfinite(r["p"]) and 0<=r["p"]<=1 for r in outerout)
    OUT.mkdir(parents=True,exist_ok=True)
    def write(name,data):
        with (OUT/name).open("w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write("c1_inner_scores.csv",innerout);write("c1_outer_oof.csv",outerout)
    write("c1_selected_parameters.csv",chosen)
    metrics=[]
    for t in taxa:
        a=[r for r in outerout if r["taxon"]==t]
        y=np.array([r["y"] for r in a]);p=np.array([r["p"] for r in a])
        eligible=len(np.unique(y))==2
        metrics.append(dict(taxon=t,n=len(y),positives=int(y.sum()),brier=float(brier_score_loss(y,p)),
            logloss=float(log_loss(y,p,labels=[0,1])),
            auc=float(roc_auc_score(y,p)) if eligible else "",
            ap=float(average_precision_score(y,p)) if eligible else "",
            auc_ap_eligible=eligible))
    write("c1_metric_eligibility.csv",metrics)
    manifest=dict(status="GLOBAL_EQUAL_TAXON_PILOT_NOT_MATCHED_10KM",outer_fold=OUTER,
       inner_spatial_groups=INNER,spatial_method="KMeans lon/lat outer-train-only; unbuffered",
       candidates=30,tuning_objective="mean over taxa of mean over inner folds of relative logloss",seed=SEED,data_sha256=sha(source),fold_sha256=sha(foldpath),
       rows=len(rows),outer_oof_rows=len(outerout),taxa=taxa,
       sklearn_version=sklearn.__version__,xgboost_version=xgboost.__version__,
       python_version=platform.python_version(),git_sha=os.environ.get("GITHUB_SHA",""),
       limitations=["not 10-km buffered","not alternative-partition matched to Bayesian/GLM",
                    "no external geographic validation","pilot fold only"])
    (OUT/"c1_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":
    main()

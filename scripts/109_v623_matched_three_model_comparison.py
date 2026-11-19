#!/usr/bin/env python3
"""Strict same-row, same-buffer XGB vs GLM comparison; descriptive paired taxon results."""
import csv,json,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,average_precision_score
X=Path("results/v623_fixed_xgb_outer_oof")
G=Path("results/v623_matched_glm_10km")
OUT=Path("results/v623_matched_three_model")
def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    gate=json.loads((X/"integrity_gate.json").read_text())
    assert gate["status"]=="PASS"
    xm=json.loads((X/"manifest.json").read_text())
    gm=json.loads((G/"manifest.json").read_text())
    for k in ("data_sha256","fold_sha256","features"):
        if xm[k]!=gm[k]:raise ValueError("Protocol manifest mismatch: "+k)
    if xm["outer_buffer_km"]!=10:raise ValueError("XGB buffer mismatch")
    xr=read(X/"outer_oof_predictions.csv")
    gr=read(G/"glm_oof_predictions.csv")
    if len(xr)!=9336 or len(gr)!=18672:raise ValueError("Unexpected row count")
    xx={r["id"]:r for r in xr}
    if len(xx)!=len(xr):raise ValueError("Duplicate XGB ID")
    models={"xgboost":xx}
    for variant in ("linear","quadratic"):
        rr=[r for r in gr if r["variant"]==variant]
        d={r["id"]:r for r in rr}
        if len(d)!=9336 or len(rr)!=len(d) or set(d)!=set(xx):raise ValueError("GLM ID coverage mismatch")
        models["glm_"+variant]=d
    for name,d in models.items():
        for id,r in d.items():
            q=xx[id]
            for k in ("plot","taxon","outer_fold","y"):
                if str(r[k])!=str(q[k]):raise ValueError(f"Mismatch {name} {id} {k}")
            p=float(r["p"])
            if not np.isfinite(p) or not 0<=p<=1:raise ValueError("Invalid prediction")
    taxa=sorted({r["taxon"] for r in xr})
    results=[]
    for taxon in taxa:
        ids=[r["id"] for r in xr if r["taxon"]==taxon]
        y=np.asarray([int(xx[id]["y"]) for id in ids])
        for name,d in models.items():
            p=np.asarray([float(d[id]["p"]) for id in ids])
            results.append(dict(taxon=taxon,model=name,n=len(ids),positives=int(y.sum()),
                logloss=float(log_loss(y,p,labels=[0,1])),
                brier=float(brier_score_loss(y,p)),
                auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p))))
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/"taxon_pooled_oof_scores.csv",results)
    contrasts=[]
    for name in ("glm_linear","glm_quadratic"):
        for metric in ("logloss","brier","auc","ap"):
            delta=np.asarray([next(float(r[metric]) for r in results if r["taxon"]==t and r["model"]=="xgboost")-
                              next(float(r[metric]) for r in results if r["taxon"]==t and r["model"]==name) for t in taxa])
            observed=float(delta.mean())
            flips=np.asarray([float(np.mean(delta*np.asarray(s))) for s in itertools.product((-1,1),repeat=len(taxa))])
            tail=float(np.mean(np.abs(flips)>=abs(observed)-1e-12))
            contrasts.append(dict(comparison="xgboost_minus_"+name,metric=metric,
                equal_taxon_mean_difference=observed,
                exact_taxon_signflip_tail_fraction=tail,
                taxa_favor_xgboost=int(np.sum(delta<0 if metric in ("logloss","brier") else delta>0)),
                n_taxa=len(taxa)))
    write(OUT/"paired_taxon_contrasts.csv",contrasts)
    summary=dict(status="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED",rows_per_model=len(xr),
        models=list(models),n_taxa=len(taxa),same_data_hash=True,same_fold_hash=True,
        comparison="Descriptive equal-taxon pooled OOF contrasts; sign-flip fractions exploratory, not confirmatory p-values.",
        caveats=["Eight taxa and spatially dependent plots limit inferential interpretation.",
                 "AUC/AP pooled per taxon are not means of fold-level metrics.",
                 "Quadratic GLM uses two prespecified squared standardized terms.",
                 "No nested hyperparameter tuning; fixed XGB versus fixed-regularization GLMs."])
    (OUT/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    print("PAIRED_CONTRASTS")
    print(json.dumps(contrasts,indent=2))
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Read-only 10-km inner spatial buffer feasibility audit; NO model training.

Inner validation groups are fixed geographically before any labels are read.
Exclude all outer-training plots within 10 km of any inner-validation plot.
Report remaining class counts, without moving partitions or using outer-test data.
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

SEED=622
INNER=4
BUFFER_KM=10.0
OUT=Path("results/v623_buffer_audit")

def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def distances_km(a,b):
    lon1,lat1=np.radians(a[:,0])[:,None],np.radians(a[:,1])[:,None]
    lon2,lat2=np.radians(b[:,0])[None,:],np.radians(b[:,1])[None,:]
    dlon, dlat=lon2-lon1,lat2-lat1
    q=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(q,0,1)))

def main():
    rows=read("data/train.csv")
    fold_rows=read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in fold_rows}
    if len(fmap)!=len(fold_rows): raise ValueError("Duplicate plot folds")
    coords={}
    outcomes={}
    for r in rows:
        p=r["plot"]
        xy=(float(r["long"]),float(r["lat"]))
        if p in coords and coords[p]!=xy: raise ValueError("Inconsistent coordinates")
        coords[p]=xy
        key=(p,r["Species"])
        if key in outcomes: raise ValueError("Duplicate taxon/plot")
        outcomes[key]=int(r["pres.abs"])
    if set(coords)!=set(fmap): raise ValueError("Plot mismatch")
    taxa=sorted({s for _,s in outcomes})
    if len(taxa)!=8 or set(fmap.values())!=set(range(1,6)):
        raise ValueError("Expected 8 taxa and 5 outer folds")
    records=[]
    exclusions=[]
    for outer in range(1,6):
        plots=sorted((p for p in fmap if fmap[p]!=outer),key=int)
        xy=np.asarray([coords[p] for p in plots])
        if not np.isfinite(xy).all(): raise ValueError("Nonfinite coordinates")
        groups=KMeans(n_clusters=INNER,random_state=SEED,n_init=10).fit_predict(xy)
        for inner in range(INNER):
            valmask=groups==inner
            valxy=xy[valmask]
            poolxy=xy[~valmask]
            poolplots=np.asarray(plots)[~valmask]
            if not len(valxy) or not len(poolxy): raise ValueError("Empty geographic group")
            nearest=distances_km(poolxy,valxy).min(axis=1)
            # Exclude training plots at or within the buffer boundary.
            retained=nearest>BUFFER_KM
            keptplots=poolplots[retained].tolist()
            valplots=np.asarray(plots)[valmask].tolist()
            for p,d in zip(poolplots,nearest):
                exclusions.append(dict(outer_fold=outer,inner_fold=inner,plot=p,
                    nearest_validation_km=float(d),retained=bool(d>BUFFER_KM)))
            for taxon in taxa:
                vp=sum(outcomes[(p,taxon)] for p in valplots)
                tp=sum(outcomes[(p,taxon)] for p in keptplots)
                n=len(keptplots)
                records.append(dict(outer_fold=outer,inner_fold=inner,taxon=taxon,
                    validation_n=len(valplots),validation_positive=vp,
                    unbuffered_train_n=len(poolplots),
                    buffered_train_n=n,buffered_train_positive=tp,
                    removed_train_n=int((~retained).sum()),
                    minimum_retained_distance_km=float(nearest[retained].min()) if n else "",
                    status="PASS" if 0<tp<n else "FAIL_SINGLE_CLASS_OR_EMPTY"))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in (("buffered_class_feasibility.csv",records),
                      ("training_plot_exclusions.csv",exclusions)):
        with (OUT/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    bad=[r for r in records if r["status"]!="PASS"]
    summary=dict(status="DIAGNOSTIC_ONLY_NO_MODEL_FITS",buffer_km=BUFFER_KM,
        buffer_rule="exclude train plots <=10 km from any inner validation plot; great-circle distance",
        partitions="frozen 4-group KMeans lon/lat, seed 622, n_init=10",
        tested_cells=len(records),failed_cells=len(bad),
        failures=bad,failures_by_outer={str(k):sum(r["outer_fold"]==k for r in bad) for k in range(1,6)},
        limitation="Inner 10-km buffer only; does not establish outer-fold buffer equivalence or valid nested tuning. No model trained.")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

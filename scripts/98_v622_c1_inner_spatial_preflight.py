#!/usr/bin/env python3
"""Outcome-blind C1 inner KMeans partition feasibility gate; no model fitting."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

OUTER=1
SEED=622
INNER=4
OUT=Path("results/v622_c1_preflight")

def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    rows=read("data/train.csv")
    folds=read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    if len(fmap)!=len(folds):
        raise ValueError("Duplicate plot fold assignment")
    plots=sorted((p for p in fmap if fmap[p]!=OUTER),key=int)
    coords={}
    for r in rows:
        if r["plot"] in plots:
            pair=(float(r["long"]),float(r["lat"]))
            if r["plot"] in coords and coords[r["plot"]]!=pair:
                raise ValueError("Inconsistent coordinates for plot "+r["plot"])
            coords[r["plot"]]=pair
    if len(coords)!=len(plots):
        raise ValueError("Missing training plot coordinates")
    xy=np.array([coords[p] for p in plots],dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite coordinates")
    km=KMeans(n_clusters=INNER,random_state=SEED,n_init=10).fit(xy)
    inner=dict(zip(plots,map(int,km.labels_)))
    if set(inner.values())!=set(range(INNER)):
        raise ValueError("Missing inner KMeans group")
    counts=defaultdict(lambda:[0,0])
    for r in rows:
        if fmap[r["plot"]]==OUTER:
            continue
        key=(r["Species"],inner[r["plot"]])
        counts[key][0]+=1
        counts[key][1]+=int(r["pres.abs"])
    taxa=sorted({r["Species"] for r in rows})
    result=[]
    for taxon in taxa:
        total_n=sum(counts[(taxon,g)][0] for g in range(INNER))
        total_p=sum(counts[(taxon,g)][1] for g in range(INNER))
        for g in range(INNER):
            n,p=counts[(taxon,g)]
            train_n=total_n-n
            train_p=total_p-p
            result.append(dict(taxon=taxon,inner_fold=g,validation_n=n,
                validation_positive=p,train_n=train_n,train_positive=train_p,
                status="PASS" if n>0 and 0<train_p<train_n else "FAIL_SINGLE_CLASS_OR_EMPTY"))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"inner_spatial_gate.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    bad=[r for r in result if r["status"]!="PASS"]
    summary=dict(status="FAIL_INNER_SPATIAL_GATE" if bad else "PASS_INNER_SPATIAL_GATE",
        outer_fold=OUTER,seed=SEED,inner_groups=INNER,method="KMeans lon/lat, n_init=10, outer training plots only",
        tested_splits=len(result),failed_splits=len(bad),failures=bad,
        limitation="unbuffered inner grouping; not matched to 10-km buffered comparator")
    (OUT/"inner_spatial_gate.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if bad:
        raise ValueError(f"{len(bad)} inner splits invalid; see inner_spatial_gate.csv")

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("C1 INNER GATE FAIL:",exc,file=sys.stderr)
        sys.exit(1)

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

def check_outer(rows,fmap,outer,taxa):
    plots=sorted((p for p in fmap if fmap[p]!=outer),key=int)
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
    positive_plots=defaultdict(list)
    for r in rows:
        if fmap[r["plot"]]==outer:
            continue
        key=(r["Species"],inner[r["plot"]])
        counts[key][0]+=1
        counts[key][1]+=int(r["pres.abs"])
        if int(r["pres.abs"])==1:
            positive_plots[key].append(r["plot"])
    result=[]
    for taxon in taxa:
        total_n=sum(counts[(taxon,g)][0] for g in range(INNER))
        total_p=sum(counts[(taxon,g)][1] for g in range(INNER))
        for g in range(INNER):
            n,p=counts[(taxon,g)]
            train_n=total_n-n
            train_p=total_p-p
            result.append(dict(outer_fold=outer,taxon=taxon,inner_fold=g,validation_n=n,
                validation_positive=p,validation_positive_plots=";".join(positive_plots[(taxon,g)]),
                train_n=train_n,train_positive=train_p,
                status="PASS" if n>0 and 0<train_p<train_n else "FAIL_SINGLE_CLASS_OR_EMPTY"))
    rare={t:{str(g):len(positive_plots[(t,g)]) for g in range(INNER)}
        for t in taxa if sum(len(positive_plots[(t,g)]) for g in range(INNER))<=10}
    return result,rare

def main():
    rows=read("data/train.csv")
    folds=read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    if len(fmap)!=len(folds):
        raise ValueError("Duplicate plot fold assignment")
    if set(fmap)!={r["plot"] for r in rows}:
        raise ValueError("Plot mismatch")
    taxa=sorted({r["Species"] for r in rows})
    if len(taxa)!=8 or set(fmap.values())!=set(range(1,6)):
        raise ValueError("Expected 8 taxa and 5 outer folds")
    all_rows=[]
    rare_by_outer={}
    for outer in range(1,6):
        result,rare=check_outer(rows,fmap,outer,taxa)
        all_rows.extend(result)
        rare_by_outer[str(outer)]=rare
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"inner_spatial_gate.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    bad=[r for r in all_rows if r["status"]!="PASS"]
    summary=dict(status="FAIL_INNER_SPATIAL_GATE" if bad else "PASS_INNER_SPATIAL_GATE",
        outer_folds=[1,2,3,4,5],seed=SEED,inner_groups=INNER,
        method="KMeans lon/lat, n_init=10, each outer training set independently",
        tested_splits=len(all_rows),failed_splits=len(bad),failures=bad,
        failures_by_outer={str(k):sum(r["outer_fold"]==k for r in bad) for k in range(1,6)},
        rare_taxon_positive_groups_by_outer=rare_by_outer,
        limitation="unbuffered inner grouping; not matched to 10-km buffered comparator")
    (OUT/"inner_spatial_gate.json").write_text(json.dumps(summary,indent=2)+"\\n")
    print(json.dumps(summary,indent=2))
    if bad:
        raise ValueError(f"{len(bad)} inner splits invalid across five outer folds; see inner_spatial_gate.csv")

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("C1 INNER GATE FAIL:",exc,file=sys.stderr)
        sys.exit(1)

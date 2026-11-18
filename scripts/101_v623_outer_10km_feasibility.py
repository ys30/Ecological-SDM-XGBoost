#!/usr/bin/env python3
"""Audit 10-km outer test/train separation for all taxa; no model fitting."""
import csv,json
from pathlib import Path
import numpy as np

OUT=Path("results/v623_outer_buffer_audit")
BUFFER_KM=10.0

def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def km(a,b):
    lo1,la1=np.radians(a[:,0])[:,None],np.radians(a[:,1])[:,None]
    lo2,la2=np.radians(b[:,0])[None,:],np.radians(b[:,1])[None,:]
    q=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(q,0,1)))

def main():
    rows=read("data/train.csv")
    folds=read("results/spatial_2d_block_assignments.csv")
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    if len(fmap)!=len(folds):raise ValueError("Duplicate plot fold")
    coords={}
    outcomes={}
    for r in rows:
        p=r["plot"];xy=(float(r["long"]),float(r["lat"]))
        if p in coords and coords[p]!=xy:raise ValueError("Inconsistent plot coordinates")
        coords[p]=xy
        k=(p,r["Species"])
        if k in outcomes:raise ValueError("Duplicate plot/taxon")
        outcomes[k]=int(r["pres.abs"])
    if set(coords)!=set(fmap):raise ValueError("Missing plot fold")
    taxa=sorted({t for _,t in outcomes})
    if len(taxa)!=8 or set(fmap.values())!=set(range(1,6)):raise ValueError("Unexpected panel")
    diagnostics=[];plot_rows=[]
    for outer in range(1,6):
        test=sorted((p for p in fmap if fmap[p]==outer),key=int)
        pool=sorted((p for p in fmap if fmap[p]!=outer),key=int)
        dist=km(np.asarray([coords[p] for p in pool]),np.asarray([coords[p] for p in test])).min(axis=1)
        kept=[p for p,d in zip(pool,dist) if d>BUFFER_KM]
        for p,d in zip(pool,dist):
            plot_rows.append(dict(outer_fold=outer,plot=p,nearest_outer_test_km=float(d),retained=bool(d>BUFFER_KM)))
        for taxon in taxa:
            trp=sum(outcomes[(p,taxon)] for p in kept)
            tep=sum(outcomes[(p,taxon)] for p in test)
            diagnostics.append(dict(outer_fold=outer,taxon=taxon,
                original_train_n=len(pool),buffered_train_n=len(kept),
                removed_train_n=len(pool)-len(kept),
                buffered_train_positive=trp,outer_test_n=len(test),
                outer_test_positive=tep,
                train_eligible=0<trp<len(kept),
                auc_ap_eligible=0<tep<len(test),
                min_retained_distance_km=float(min((d for d in dist if d>BUFFER_KM),default=float("nan")))))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in (("outer_buffer_feasibility.csv",diagnostics),("outer_training_plot_exclusions.csv",plot_rows)):
        with (OUT/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    bad=[r for r in diagnostics if not r["train_eligible"]]
    summary=dict(status="NO_FITS_OUTER_BUFFER_AUDIT",buffer_km=BUFFER_KM,
        tested_outer_taxon_cells=len(diagnostics),ineligible_train_cells=len(bad),
        ineligible=bad,auc_ap_ineligible_cells=sum(not r["auc_ap_eligible"] for r in diagnostics),
        caveat="Outer buffer feasibility only. It does not validate inner tuning or solve one-positive training cases.")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

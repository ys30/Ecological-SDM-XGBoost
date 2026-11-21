#!/usr/bin/env python3
"""No-fit spatial inner-validation feasibility matrix for v6.24 amendment.

Outcome-blind partitions: contiguous quantile cells from outer-training
coordinates only; class counts inspected only AFTER partitions are frozen.
This is a diagnostic, not an approved tuning split or nested model fit.
"""
import csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
D=Path("results/v624_inner_feasibility");D.mkdir(parents=True,exist_ok=True)
def read(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
rows=read("data/train.csv");folds=read("results/spatial_2d_block_assignments.csv")
fmap={r["plot"]:int(r["fold"]) for r in folds}
taxa=sorted({r["Species"] for r in rows})
assert len(taxa)==8 and len(fmap)==1167
coords={}
for r in rows:
    p=r["plot"];xy=(float(r["long"]),float(r["lat"]))
    if p in coords:assert coords[p]==xy
    coords[p]=xy
def quantile(a,q):
    z=sorted(a);pos=(len(z)-1)*q;lo=int(pos);hi=min(lo+1,len(z)-1)
    return z[lo]*(1-(pos-lo))+z[hi]*(pos-lo)
report=[];impossible=[]
for outer in range(1,6):
    outer_plots=sorted((p for p in coords if fmap[p]!=outer),key=int)
    x=[coords[p][0] for p in outer_plots];y=[coords[p][1] for p in outer_plots]
    xcut=[quantile(x,q/4) for q in (1,2,3)]
    ycut=[quantile(y,q/4) for q in (1,2,3)]
    groups=defaultdict(list)
    for p in outer_plots:
        xx,yy=coords[p]
        ix=sum(xx>c for c in xcut);iy=sum(yy>c for c in ycut)
        groups[(ix,iy)].append(p)
    for t in taxa:
        outcomes={r["plot"]:int(r["pres.abs"]) for r in rows if r["Species"]==t}
        total=sum(outcomes[p] for p in outer_plots)
        if total<2:impossible.append({"outer_fold":outer,"taxon":t,"outer_training_positives":total,"reason":"cannot validate a positive and retain a training positive"})
        for (ix,iy),val in sorted(groups.items()):
            vpos=sum(outcomes[p] for p in val);tpos=total-vpos
            report.append(dict(outer_fold=outer,taxon=t,cell_x=ix,cell_y=iy,
                training_n=len(outer_plots)-len(val),validation_n=len(val),
                training_positives=tpos,validation_positives=vpos,
                feasible_training=int(tpos>0 and tpos<len(outer_plots)-len(val)),
                informative_validation=int(vpos>0 and vpos<len(val)),
                status="PASS" if 0<tpos<len(outer_plots)-len(val) else "FAIL_SINGLE_CLASS"))
assert report
with (D/"inner_cell_feasibility.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
summary=dict(status="NO_FIT_DIAGNOSTIC_ONLY",outer_folds=5,taxa=len(taxa),
    cell_rows=len(report),invalid_training_cells=sum(r["status"]!="PASS" for r in report),
    informative_validation_cells=sum(r["informative_validation"] for r in report),
    mathematical_impossibilities=impossible,
    partitions="outer-training-only 4x4 longitude/latitude quantile cells",
    caution="not distance-buffered; no hyperparameter fits; not an approved inner CV",
    data_sha256=hashlib.sha256(Path("data/train.csv").read_bytes()).hexdigest(),
    fold_sha256=hashlib.sha256(Path("results/spatial_2d_block_assignments.csv").read_bytes()).hexdigest())
(D/"feasibility_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))

#!/usr/bin/env python3
"""Held-out distance-stratified ROC AUC/AP from archived matched OOF predictions.

No model refit. Distances use WGS84-like lon/lat columns and haversine km;
verify source CRS before treating them as authoritative geodesic distances.
"""
import csv, math
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):
    with (R/p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
a=read("results/spatial_2d_block_assignments.csv")
b=read("results/bayes_spatial_cv_predictions.csv")
m=read("results/maxent_spatial_2d_predictions.csv")
x=read("results/spatial_2d_block_cv_predictions.csv")
assert len(a)==1167 and len(b)==len(x)==9336
loc={z["plot"]:z for z in a}
assert len(loc)==len(a)
def key(z):return z["id"],z["plot"],z["fold"],z["observed"]
bi={key(z):z for z in b}; xi={key(z):z for z in x}; mi={key(z):z for z in m}
assert len(bi)==len(b) and len(xi)==len(x) and len(mi)==len(m)
assert set(bi)==set(xi) and set(mi).issubset(bi)
coords={p:(math.radians(float(z["lat"])),math.radians(float(z["long"]))) for p,z in loc.items()}
def km(u,v):
    p,q=coords[u],coords[v];h=math.sin((p[0]-q[0])/2)**2+math.cos(p[0])*math.cos(q[0])*math.sin((p[1]-q[1])/2)**2
    return 12742.0176*math.asin(min(1,math.sqrt(max(0,h))))
distance={}
for fold in range(1,6):
    tr=[p for p,z in loc.items() if int(z["fold"])!=fold]
    te=[p for p,z in loc.items() if int(z["fold"])==fold]
    for p in te:distance[p]=min(km(p,q) for q in tr)
assert len(distance)==1167
def auc(y,p):
    pos=sum(y);neg=len(y)-pos
    if not pos or not neg:return None
    v=sorted(zip(p,y));i=0;rs=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[j][0]==v[i][0]:j+=1
        rs+=(i+1+j)/2*sum(t for _,t in v[i:j]);i=j
    return (rs-pos*(pos+1)/2)/(pos*neg)
def ap(y,p):
    """Threshold-grouped AP, invariant to ordering of equal scores."""
    if not sum(y):return None
    v=sorted(zip(p,y),reverse=True);tp=0;seen=0;area=0.;i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[j][0]==v[i][0]:j+=1
        newly=sum(t for _,t in v[i:j]);tp+=newly;seen+=j-i
        area+=newly*tp/seen;i=j
    return area/sum(y)
groups=defaultdict(list)
for k,z in mi.items():
    sp=z["species"];fold=int(z["fold"]);p=z["plot"];dist=distance[p]
    y=int(z["observed"])
    vals={"bayesian":float(bi[k]["mean_probability"]),"maxent":float(z["relative_suitability"]),"xgboost":float(xi[k]["predicted_probability"])}
    for band,keep in [("all",True),("ge5km",dist>=5),("lt5km",dist<5),("ge1km",dist>=1)]:
        if keep:groups[(sp,fold,band)].append((y,vals))
out=[]
for (sp,fold,band),records in sorted(groups.items()):
    y=[v[0] for v in records]; row=dict(species=sp,fold=fold,distance_band=band,n=len(y),positives=sum(y),\n        auc_status="ok" if 0<sum(y)<len(y) else "undefined_single_class")
    for model in ("bayesian","maxent","xgboost"):
        scores=[v[1][model] for v in records]
        row[model+"_auc"]=auc(y,scores);row[model+"_ap"]=ap(y,scores)
    out.append(row)
dest=R/"results"/"matched_distance_stratified_oof_metrics.csv"
with dest.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print("Distance bands by held-out plot:",{s:sum((d<5 if s=="lt5km" else d>=5) for d in distance.values()) for s in ("lt5km","ge5km")})
print("Saved",len(out),"rows to",dest)
print("IMPORTANT: subsets with no positive/negative labels have undefined AUC.")

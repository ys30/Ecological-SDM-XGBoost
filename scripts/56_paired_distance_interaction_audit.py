#!/usr/bin/env python3
"""Paired near/far diagnostic for matched OOF predictions (no refitting).

Five taxa, overlapping spatial folds: sign-flip is descriptive sensitivity only.
"""
import csv,itertools,statistics
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1]
with (R/"results/matched_distance_stratified_oof_metrics.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
idx={(r["species"],r["fold"],r["distance_band"]):r for r in rows}
out=[]
for (sp,fold,band),near in sorted(idx.items()):
    if band!="lt5km" or near["auc_status"]!="ok":continue
    far=idx[(sp,fold,"ge5km")]
    if far["auc_status"]!="ok":continue
    d=dict(species=sp,fold=fold,near_n=int(near["n"]),far_n=int(far["n"]),
        near_positives=int(near["positives"]),far_positives=int(far["positives"]))
    for metric in ("auc","ap"):
        for model in ("bayesian","maxent","xgboost"):
            d[f"{model}_{metric}_near_minus_far"]=float(near[f"{model}_{metric}"])-float(far[f"{model}_{metric}"])
        for model in ("xgboost","bayesian"):
            d[f"{model}_minus_maxent_{metric}_interaction"]=(
                float(near[f"{model}_{metric}"])-float(near[f"maxent_{metric}"])
                -float(far[f"{model}_{metric}"])+float(far[f"maxent_{metric}"]))
    out.append(d)
assert len(out)==21 and len({r["species"] for r in out})==5
dest=R/"results/paired_distance_interaction_audit.csv"
with dest.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
summ=[]
for metric in ("auc","ap"):
    for model in ("xgboost","bayesian"):
        k=f"{model}_minus_maxent_{metric}_interaction"
        vals=[r[k] for r in out]
        taxa=defaultdict(list)
        for r in out:taxa[r["species"]].append(r[k])
        taxon_means=[statistics.mean(v) for v in taxa.values()]
        observed=statistics.mean(taxon_means)
        null=[statistics.mean(sign*v for sign,v in zip(signs,taxon_means))
              for signs in itertools.product((-1,1),repeat=len(taxon_means))]
        p=sum(abs(v)>=abs(observed)-1e-12 for v in null)/len(null)
        summ.append(dict(contrast=k,eligible_cells=len(out),taxa=len(taxa),
            cell_mean=statistics.mean(vals),equal_taxon_mean=observed,
            exploratory_exact_taxon_signflip_p=p,
            taxon_means=";".join(f"{s}:{statistics.mean(v):.6f}" for s,v in sorted(taxa.items()))))
with (R/"results/paired_distance_interaction_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
for r in summ:print(r)

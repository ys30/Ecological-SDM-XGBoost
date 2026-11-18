#!/usr/bin/env python3
"""Descriptive taxon-level calibration diagnostics; no retraining or posthoc calibration."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss
BASE=Path("results/v623_fixed_xgb_outer_oof")
def read(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    gate=json.loads((BASE/"integrity_gate.json").read_text())
    if gate.get("status")!="PASS":raise ValueError("OOF integrity gate required")
    pred=read(BASE/"outer_oof_predictions.csv")
    taxa=sorted({r["taxon"] for r in pred})
    overall=[];bins=[]
    for taxon in taxa:
        rows=[r for r in pred if r["taxon"]==taxon]
        y=np.asarray([int(r["y"]) for r in rows],dtype=float)
        p=np.asarray([float(r["p"]) for r in rows])
        overall.append(dict(taxon=taxon,n=len(rows),observed_prevalence=float(y.mean()),
            mean_predicted_probability=float(p.mean()),calibration_in_the_large=float(p.mean()-y.mean()),
            brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p,labels=[0,1]))))
        # Equal-count ranks within each taxon; not a trained calibration model.
        for bin_index,indices in enumerate(np.array_split(np.argsort(p,kind="stable"),10),start=1):
            bins.append(dict(taxon=taxon,decile=bin_index,n=len(indices),
                mean_predicted=float(p[indices].mean()),observed_fraction=float(y[indices].mean()),
                positives=int(y[indices].sum())))
    write(BASE/"taxon_calibration_in_the_large.csv",overall)
    write(BASE/"taxon_reliability_deciles.csv",bins)
    print(json.dumps(dict(status="DESCRIPTIVE_ONLY",taxa=len(taxa),reliability_bins=len(bins),
        note="Decile observed fractions for rare taxa are unstable; no fitted calibration correction or confidence intervals."),indent=2))
if __name__=="__main__":main()

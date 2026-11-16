#!/usr/bin/env python3
"""Read-only v6.22 C1 canonical input/outer fold preflight; no model fitting."""
import csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

DATA=Path("data/train.csv")
FOLDS=Path("results/spatial_2d_block_assignments.csv")
OUT=Path("results/v622_c1_preflight")
def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fail(msg): raise ValueError(msg)
def main():
    d=read(DATA); f=read(FOLDS)
    if not d or not f: fail("Empty source")
    if len({r["id"] for r in d})!=len(d): fail("Duplicate observation IDs")
    fmap={}
    for r in f:
        if r["plot"] in fmap: fail("Duplicate plot in fold file")
        fmap[r["plot"]]=r["fold"]
    if set(fmap)!={r["plot"] for r in d}: fail("Plot mismatch between data and folds")
    if set(fmap.values())!={"1","2","3","4","5"}: fail("Expected five spatial folds")
    taxa=sorted({r["Species"] for r in d})
    if len(taxa)!=8: fail("Expected eight anonymous taxa")
    perplot=defaultdict(set); cells=Counter(); labels=Counter()
    for r in d:
        if r["pres.abs"] not in ("0","1"): fail("Nonbinary outcome")
        if any(not r[k] for k in ("rainann","soildepth","tempann","topo")): fail("Missing predictor")
        for k in ("rainann","soildepth","tempann","topo"):
            v=float(r[k])
            if not __import__("math").isfinite(v): fail("Nonfinite predictor")
        perplot[r["plot"]].add(r["Species"])
        cells[(r["Species"],fmap[r["plot"]])]+=1
        labels[(r["Species"],fmap[r["plot"]])]+=int(r["pres.abs"])
    if any(v!=set(taxa) for v in perplot.values()): fail("Incomplete taxon panel")
    if len(cells)!=40: fail("Missing taxon-fold cells")
    # Check outer-training class feasibility across all five spatial folds.
    outer_train=[]
    for s in taxa:
        total_n=sum(cells[(s,k)] for k in ("1","2","3","4","5"))
        total_pos=sum(labels[(s,k)] for k in ("1","2","3","4","5"))
        for k in ("1","2","3","4","5"):
            n=total_n-cells[(s,k)]
            pos=total_pos-labels[(s,k)]
            outer_train.append({"taxon":s,"outer_fold":k,"train_n":n,
                "train_positive":pos,"test_n":cells[(s,k)],
                "test_positive":labels[(s,k)],
                "train_binary":0<pos<n,
                "test_auc_ap_eligible":0<labels[(s,k)]<cells[(s,k)]})
    if any(not z["train_binary"] for z in outer_train):
        fail("At least one outer training fold single-class")
    summary={"status":"PASS_INPUT_AND_OUTER_FOLDS_ONLY","data_rows":len(d),"plots":len(perplot),"taxa":taxa,
      "data_sha256":sha(DATA),"folds_sha256":sha(FOLDS),"fold_sizes":dict(Counter(fmap.values())),
      "outer_training_diagnostics":outer_train,
      "taxon_fold":[{"taxon":s,"fold":k,"n":cells[(s,k)],"positive":labels[(s,k)],
      "auc_ap_eligible":0<labels[(s,k)]<cells[(s,k)]} for s in taxa for k in sorted(set(fmap.values()))],
      "not_checked":["inner folds","buffer CRS","30 candidate tuning","model fit","outer OOF","CI model success"]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"input_gate.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="taxon_fold"},indent=2))
if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("C1 PREFLIGHT FAIL:",exc,file=sys.stderr)
        sys.exit(1)

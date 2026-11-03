#!/usr/bin/env python3
"""Reusable matched out-of-fold binary-model evaluation (standard library only).

Input: one CSV with observation_id, taxon, fold, observed, and any score columns.
For each taxon-fold, compare models only on the *intersection* of nonmissing scores.
AUC uses midranks for tied scores; AP groups tied scores at one threshold.
This evaluates rankings, not calibration or causal algorithm effects.

Example:
  python scripts/52_evaluate_matched_oof.py --input joined.csv \
    --scores bayesian maxent xgboost --output results/evaluation.csv
"""
import argparse,csv,math
from collections import defaultdict
from pathlib import Path

def auc(y,s):
    n1=sum(y);n0=len(y)-n1
    if not n1 or not n0:return float("nan")
    z=sorted(zip(s,y));i=0;ranks=0.
    while i<len(z):
        j=i+1
        while j<len(z) and z[j][0]==z[i][0]:j+=1
        ranks+=sum(t for _,t in z[i:j])*(i+1+j)/2
        i=j
    return (ranks-n1*(n1+1)/2)/(n1*n0)

def ap(y,s):
    if not sum(y):return float("nan")
    z=sorted(zip(s,y),reverse=True);i=0;seen=0;hits=0;area=0.
    while i<len(z):
        j=i+1
        while j<len(z) and z[j][0]==z[i][0]:j+=1
        newhits=sum(t for _,t in z[i:j])
        seen+=j-i;hits+=newhits
        area+=newhits*hits/seen
        i=j
    return area/sum(y)

def evaluate(records,scores):
    groups=defaultdict(list)
    ids=set()
    for r in records:
        k=(r["taxon"],r["fold"],r["observation_id"])
        if k in ids:raise ValueError(f"duplicate key {k}")
        ids.add(k)
        if r["observed"] not in ("0","1"):raise ValueError(f"nonbinary label {k}")
        groups[(r["taxon"],r["fold"])].append(r)
    out=[]
    for (taxon,fold),group in sorted(groups.items()):
        valid=[]
        for r in group:
            try:values=[float(r[s]) for s in scores]
            except (ValueError,KeyError,TypeError):continue
            if all(math.isfinite(x) for x in values):valid.append((int(r["observed"]),values))
        y=[z[0] for z in valid]
        row=dict(taxon=taxon,fold=fold,n_all=len(group),n_matched=len(valid),
                 positives=sum(y),status="ok" if y and sum(y)<len(y) else "insufficient_classes")
        for i,model in enumerate(scores):
            s=[z[1][i] for z in valid]
            row[model+"_auc"]=auc(y,s) if row["status"]=="ok" else float("nan")
            row[model+"_ap"]=ap(y,s) if y else float("nan")
        out.append(row)
    return out

def selftest():
    assert math.isclose(auc([0,1,0,1],[.1,.9,.2,.8]),1.)
    assert math.isclose(ap([0,1,0,1],[.1,.9,.2,.8]),1.)
    # One positive and one negative at the same score => grouped AP = prevalence
    assert math.isclose(auc([1,0],[.5,.5]),.5)
    assert math.isclose(ap([1,0],[.5,.5]),.5)
    assert math.isnan(auc([0,0],[.2,.3]))
    rows=[dict(observation_id=str(i),taxon="a",fold="1",observed=str(y),
               m=str(s),n=str(s)) for i,(y,s) in enumerate(zip([1,0,1],[.7,.7,.2]))]
    out=evaluate(rows,["m","n"])
    assert out[0]["n_matched"]==3 and out[0]["status"]=="ok"
    assert out[0]["m_auc"]==out[0]["n_auc"]
    print("PASS: reusable evaluator self-tests")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path)
    p.add_argument("--scores",nargs="+")
    p.add_argument("--output",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:selftest();return
    if not (args.input and args.output and args.scores):p.error("--input, --scores and --output required")
    if len(set(args.scores))!=len(args.scores):p.error("score names must be unique")
    with args.input.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        required={"observation_id","taxon","fold","observed",*args.scores}
        if not required.issubset(reader.fieldnames or []):p.error(f"missing columns: {required-set(reader.fieldnames or [])}")
        result=evaluate(reader,args.scores)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    print("Evaluated",len(result),"taxon-fold cells; matched score columns:",", ".join(args.scores))
if __name__=="__main__":main()

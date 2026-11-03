#!/usr/bin/env python3
"""Matched retrospective top-k retrieval comparison: Bayesian vs common-input XGBoost.

Uses identical anonymous taxon, held-out fold, sample count, positive count and
per-fold budgets. Counts are aggregated across folds; not independent trials.
"""
import csv
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parents[1]
def read(p):
    with (R/p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
bay=read("results/bayesian_rare_species_within_fold_retrieval.csv")
xgb=read("results/v614_common_input/rare_taxon_within_fold_retrieval.csv")
bi={(r["species"],int(r["fold"])):r for r in bay}
xi={(r["species"],int(r["fold"]),int(r["budget"])):r for r in xgb}
assert len(bi)==15 and len(xi)==45
rows=[]
for sp in ("nsw19","nsw20","nsw22"):
    for k in (10,25,50):
        a=b=total=expected=0
        for fold in range(1,6):
            z=bi[(sp,fold)];x=xi[(sp,fold,k)]
            assert x["status"]=="ok"
            assert int(z["n"])==int(x["n"]) and int(z["positives"])==int(x["positives"])
            p=int(z["positives"]);n=int(z["n"])
            assert n>=k
            total+=p;expected+=p*k/n
            aa=int(z[f"top_{k}_captured"]);bb=int(x["captured"])
            assert 0<=aa<=p and 0<=bb<=p
            a+=aa;b+=bb
        rows.append(dict(species=sp,budget_per_fold=k,total_inspected=5*k,
            positives=total,bayesian_captured=a,xgboost_captured=b,
            bayesian_recall=a/total,xgboost_recall=b/total,
            delta_xgboost_minus_bayesian_recall=(b-a)/total,
            random_expected=expected))
out=R/"results"/"rare_taxon_bayesian_xgboost_matched_topk.csv"
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(r["species"],"top",r["budget_per_fold"],"Bayes",r["bayesian_captured"],"XGB",r["xgboost_captured"],"of",r["positives"])
# Standalone publication-friendly SVG: compact paired recall by budget and taxon.
W,H=930,450
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Bayesian and XGBoost rare taxon retrieval recall by inspection budget">',
'<rect width="930" height="450" fill="white"/>',
'<text x="44" y="36" font-family="Arial,sans-serif" font-size="21" font-weight="bold" fill="#172b4d">Rare-taxon retrieval: matched spatial folds</text>',
'<text x="44" y="59" font-family="Arial,sans-serif" font-size="12" fill="#526579">Recall among observed presences; top-k plots selected independently in each of five folds</text>']
for j,sp in enumerate(("nsw19","nsw20","nsw22")):
    left=58+j*305
    parts.append(f'<text x="{left}" y="99" font-family="Arial,sans-serif" font-size="16" font-weight="bold" fill="#172b4d">{sp}</text>')
    for t in (0,.5,1):
        yy=340-200*t
        parts.append(f'<line x1="{left+30}" y1="{yy}" x2="{left+255}" y2="{yy}" stroke="#e3e8ee"/>')
        parts.append(f'<text x="{left+22}" y="{yy+4}" font-family="Arial,sans-serif" font-size="10" text-anchor="end" fill="#6b7280">{t:g}</text>')
    for i,k in enumerate((10,25,50)):
        r=next(v for v in rows if v["species"]==sp and v["budget_per_fold"]==k)
        xx=left+67+i*70
        for offset,model,color in ((-11,"bayesian","#3269a8"),(3,"xgboost","#db7b30")):
            value=r[model+"_recall"];height=200*value
            parts.append(f'<rect x="{xx+offset}" y="{340-height:.2f}" width="13" height="{height:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{xx}" y="359" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#344054">{k}</text>')
    parts.append(f'<text x="{left+145}" y="381" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#526579">Top-k per fold</text>')
parts.extend(['<rect x="345" y="412" width="13" height="13" fill="#3269a8"/>','<text x="363" y="423" font-family="Arial,sans-serif" font-size="12" fill="#344054">Bayesian</text>','<rect x="450" y="412" width="13" height="13" fill="#db7b30"/>','<text x="468" y="423" font-family="Arial,sans-serif" font-size="12" fill="#344054">Taxon-specific XGBoost</text>','</svg>'])
fig=R/"results"/"figures"/"27_rare_taxon_matched_topk_bayesian_xgboost.svg"
fig.parent.mkdir(parents=True,exist_ok=True)
fig.write_text("\n".join(parts),encoding="utf-8")
print("Saved",out,fig)

import csv, math, pathlib
root=pathlib.Path("results")
rows=[]
for fold in range(1,6):
    path=root/f"bayes_spatial_fold_{fold}_diagnostics.csv"
    with path.open(newline="") as f:
        data=list(csv.DictReader(f))
    for row in data:
        rhat=float(row["rhat"]); ess=float(row["ess_bulk"])
        rows.append(dict(fold=fold,variable=row["variable"],rhat=rhat,ess_bulk=ess,
                         rhat_above_101=int(not math.isfinite(rhat) or rhat>1.01),
                         ess_below_400=int(not math.isfinite(ess) or ess<400)))
with (root/"bayes_mcmc_diagnostics_all.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for fold in range(1,6):
    v=[r for r in rows if r["fold"]==fold]
    summary.append(dict(fold=fold,parameters=len(v),max_rhat=max(r["rhat"] for r in v),
        min_bulk_ess=min(r["ess_bulk"] for r in v),
        rhat_above_101=sum(r["rhat_above_101"] for r in v),
        ess_below_400=sum(r["ess_below_400"] for r in v)))
with (root/"bayes_mcmc_diagnostics_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
for row in summary: print(row)
print("NOTE: Divergences, treedepth and BFMI are not available in the exported diagnostics; these checks remain pending.")

"""Audit held-out Bayesian posterior uncertainty for rare tree species."""
import csv, math, pathlib, statistics
root=pathlib.Path("results")
with (root/"bayes_spatial_cv_predictions.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
species=sorted({r["species"] for r in rows})
output=[]
for sp in species:
    subset=[r for r in rows if r["species"]==sp]
    p=[float(r["mean_probability"]) for r in subset]
    lo=[float(r["lower_95"]) for r in subset]
    hi=[float(r["upper_95"]) for r in subset]
    y=[int(r["observed"]) for r in subset]
    assert all(0<=a<=b<=1 for a,b in zip(lo,hi))
    assert all(a<=b<=c for a,b,c in zip(lo,p,hi))
    output.append(dict(species=sp,n=len(subset),presences=sum(y),prevalence=sum(y)/len(y),
        mean_prediction=statistics.mean(p),mean_interval_width=statistics.mean(b-a for a,b in zip(lo,hi)),
        median_interval_width=statistics.median(b-a for a,b in zip(lo,hi)),
        mean_width_present=statistics.mean([hi[i]-lo[i] for i in range(len(y)) if y[i]]) if sum(y) else "",
        mean_width_absent=statistics.mean([hi[i]-lo[i] for i in range(len(y)) if not y[i]]) if sum(y)<len(y) else ""))
with (root/"bayes_species_uncertainty_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]))
    w.writeheader();w.writerows(output)
for row in output: print(row)
print("Intervals are posterior predictive mean-probability credible intervals, not coverage-calibrated frequentist confidence intervals.")

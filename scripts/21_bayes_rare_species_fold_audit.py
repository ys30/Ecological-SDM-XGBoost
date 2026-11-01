"""Summarize spatial-fold uncertainty for rare species from held-out predictions."""
import csv, statistics
from pathlib import Path
source=Path("results/bayes_spatial_cv_predictions.csv")
with source.open(newline="") as f: rows=list(csv.DictReader(f))
rare={"Corymbia.gummifera","Corymbia.intermedia","Eucalyptus.carnea"}
out=[]
for species in sorted(rare):
    for fold in sorted({int(r["fold"]) for r in rows}):
        group=[r for r in rows if r["species"]==species and int(r["fold"])==fold]
        if not group: continue
        positives=[r for r in group if int(r["observed"])==1]
        def mean_width(items):
            return statistics.mean(float(r["upper_95"])-float(r["lower_95"]) for r in items) if items else ""
        out.append(dict(species=species,fold=fold,n=len(group),presences=len(positives),
            mean_prediction=statistics.mean(float(r["mean_probability"]) for r in group),
            mean_interval_width=mean_width(group),mean_width_present=mean_width(positives),
            positive_fraction=len(positives)/len(group)))
path=Path("results/bayes_rare_species_fold_uncertainty.csv")
with path.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for row in out: print(row)
print("Fold-level summaries are descriptive; tiny positive counts limit interpretation.")

"""Descriptive tuning x sharing interaction, paired at taxon-fold level.

Interaction = (tuned-fixed) species - (tuned-fixed) pooled.
AUC positive and proper-score negative favor species-specific tuning.
Bootstrap resamples taxa, preserving both variants' folds; only 8 clusters.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("results/v622")
METRICS=("auc","logloss","brier")
def main():
    df=pd.read_csv(ROOT/"c1_fixed_vs_tuned_paired_cells.csv")
    keys=["design","Species","outer_fold"]
    a=df.loc[df.variant=="species"].drop(columns=["variant"])
    b=df.loc[df.variant=="pooled"].drop(columns=["variant"])
    j=a.merge(b,on=keys,how="outer",indicator=True,validate="one_to_one",suffixes=("_species","_pooled"))
    if len(j)!=80 or not (j._merge=="both").all():raise ValueError("expected complete paired taxon-fold coverage")
    taxa=np.array(sorted(j.Species.unique()))
    if len(taxa)!=8:raise ValueError("expected eight taxa")
    rng=np.random.default_rng(622)
    samples=rng.choice(taxa,size=(2000,8),replace=True)
    result={"status":"DESCRIPTIVE_TUNING_SHARING_INTERACTION_NOT_CONFIRMATORY",
      "definition":"(species tuned - species fixed) - (pooled tuned - pooled fixed)",
      "bootstrap":"2000 paired taxon resamples seed 622; same sampled taxa for both spatial designs",
      "limitations":["Only 8 taxa","No multiple-testing correction","R-original vs Python-fixed implementation equivalence unverified",
                     "AUC excludes non-evaluable folds; proper scores include fallback records"],
      "designs":{}}
    for d,g in j.groupby("design"):
        rec={"cells":len(g),"metrics":{}}
        for metric in METRICS:
            col="interaction_"+metric
            g=g.copy()
            g[col]=g[f"delta_{metric}_tuned_minus_fixed_species"]-g[f"delta_{metric}_tuned_minus_fixed_pooled"]
            values=g[col].dropna()
            groups={t:g.loc[g.Species==t,col].dropna().to_numpy(float) for t in taxa}
            draws=[]
            for sample in samples:
                arrays=[groups[t] for t in sample if len(groups[t])]
                if not arrays:raise ValueError("bootstrap sample lacks evaluable AUC")
                draws.append(float(np.concatenate(arrays).mean()))
            ci=np.quantile(draws,[.025,.975])
            rec["metrics"][metric]={"mean":float(values.mean()),"ci95":[float(ci[0]),float(ci[1])],
               "evaluable_cells":int(len(values))}
        result["designs"][d]=rec
    (ROOT/"c1_tuning_sharing_interaction.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

"""Taxon-cluster paired bootstrap for C1 fixed-vs-tuned contrasts.
Descriptive uncertainty; eight taxon clusters and no multiplicity adjustment.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path("results/v622")
METRICS=["auc","logloss","brier"]
def main():
    df=pd.read_csv(ROOT/"c1_fixed_vs_tuned_paired_cells.csv")
    if len(df)!=160 or df.Species.nunique()!=8:raise ValueError("invalid paired table")
    taxa=np.array(sorted(df.Species.unique()))
    rng=np.random.default_rng(622)
    samples=rng.choice(taxa,size=(2000,len(taxa)),replace=True)
    results={}
    for (design,variant),g in df.groupby(["design","variant"]):
        key=design+"/"+variant
        results[key]={"cells":len(g),"metrics":{}}
        for metric in METRICS:
            col="delta_"+metric+"_tuned_minus_fixed"
            values=g[col].dropna().to_numpy(float)
            groups={t:g.loc[g.Species==t,col].dropna().to_numpy(float) for t in taxa}
            draws=[]
            for sample in samples:
                selected=[groups[t] for t in sample if len(groups[t])]
                if not selected:raise ValueError("no valid cells in bootstrap replicate")
                draws.append(float(np.concatenate(selected).mean()))
            arr=np.asarray(draws)
            ci=np.quantile(arr,[.025,.975])
            results[key]["metrics"][metric]={"mean":float(values.mean()),"ci95":[float(ci[0]),float(ci[1])],
                "evaluable_cells":int(len(values))}
    output={"status":"DESCRIPTIVE_C1A_C1B_CLUSTER_BOOTSTRAP","seed":622,"draws":2000,
      "resampling_unit":"taxon, preserving all available outer folds",
      "interpretation":"No multiplicity adjustment, only 8 taxa, fixed arms Python reimplementations of historical R settings",
      "results":results}
    (ROOT/"c1_fixed_vs_tuned_bootstrap.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)
if __name__=="__main__":main()

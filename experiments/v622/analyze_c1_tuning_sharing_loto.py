"""Leave-one-taxon-out sensitivity for C1 tuning x sharing interaction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path("results/v622")
def main():
    df=pd.read_csv(ROOT/"c1_fixed_vs_tuned_paired_cells.csv")
    keys=["design","Species","outer_fold"]
    a=df[df.variant=="species"].drop(columns="variant")
    b=df[df.variant=="pooled"].drop(columns="variant")
    j=a.merge(b,on=keys,validate="one_to_one",suffixes=("_species","_pooled"))
    if len(j)!=80 or j.Species.nunique()!=8:raise ValueError("incomplete paired interaction")
    result={"status":"DESCRIPTIVE_LOTO","definition":"species tuning effect minus pooled tuning effect","designs":{}}
    for design,g in j.groupby("design"):
        output={}
        for metric in ("auc","logloss","brier"):
            x=g.assign(interaction=g[f"delta_{metric}_tuned_minus_fixed_species"]-g[f"delta_{metric}_tuned_minus_fixed_pooled"])
            overall=float(x.interaction.mean())
            loo={str(t):float(x.loc[x.Species!=t,"interaction"].mean()) for t in sorted(x.Species.unique())}
            output[metric]={"full":overall,"leave_one_taxon_out":loo,
                "range":[float(min(loo.values())),float(max(loo.values()))],
                "sign_stable":all(np.sign(v)==np.sign(overall) for v in loo.values())}
        result["designs"][design]=output
    path=ROOT/"c1_tuning_sharing_loto.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

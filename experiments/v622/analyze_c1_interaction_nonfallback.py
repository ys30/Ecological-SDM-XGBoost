"""C1 tuning-sharing interaction sensitivity excluding all fallback predictions.

All four arms aligned by plot, Species and outer_fold. Keep only records for
which no arm used fallback. Compute log loss and Brier per taxon-fold, then
compare the two tuned-minus-fixed differences on identical retained rows.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

ROOT=Path("results/v622")
def main():
    result={"status":"DESCRIPTIVE_NONFALLBACK_SENSITIVITY",
      "population":"records with fallback=0 in all four arms, within each spatial design",
      "estimand":"equal-weight taxon-fold mean of species tuning gain minus pooled tuning gain",
      "designs":{}}
    for design in ("grid4_s2026_b0","km15_b0"):
        keys=["plot","Species","outer_fold"]
        merged=None
        for variant in ("species","pooled"):
            for setting in ("tuned","fixed"):
                p=ROOT/f"oof_xgb_{variant}_{setting}_{design}.csv.gz"
                df=pd.read_csv(p)
                arm=f"{variant}_{setting}"
                df=df[keys+["pres.abs","pred","fallback"]].rename(columns={"pres.abs":f"y_{arm}","pred":f"p_{arm}","fallback":f"f_{arm}"})
                merged=df if merged is None else merged.merge(df,on=keys,how="outer",validate="one_to_one",indicator=False)
        if len(merged)!=9336 or merged[keys].isna().any().any():raise ValueError("OOF join mismatch")
        arms=["species_tuned","species_fixed","pooled_tuned","pooled_fixed"]
        for arm in arms[1:]:
            if not (merged["y_"+arm]==merged["y_"+arms[0]]).all():raise ValueError("label mismatch")
        eligible=np.ones(len(merged),dtype=bool)
        for arm in arms:eligible &= merged["f_"+arm].to_numpy()==0
        sub=merged.loc[eligible]
        rows=[]
        for (taxon,fold),g in sub.groupby(["Species","outer_fold"]):
            y=g["y_species_tuned"].to_numpy(int)
            rec={"Species":taxon,"fold":int(fold),"n":len(g)}
            for metric in ("logloss","brier"):
                scores={}
                for arm in arms:
                    p=np.clip(g["p_"+arm].to_numpy(float),1e-6,1-1e-6)
                    scores[arm]=float(log_loss(y,p,labels=[0,1])) if metric=="logloss" else float(brier_score_loss(y,p))
                rec[metric]=(scores["species_tuned"]-scores["species_fixed"])-(scores["pooled_tuned"]-scores["pooled_fixed"])
            rows.append(rec)
        cells=pd.DataFrame(rows)
        if len(cells)==0:raise ValueError("no eligible cells")
        taxa=np.array(sorted(merged.Species.unique()))
        rng=np.random.default_rng(622)
        samples=rng.choice(taxa,size=(2000,len(taxa)),replace=True)
        metrics={}
        for metric in ("logloss","brier"):
            groups={t:cells.loc[cells.Species==t,metric].to_numpy(float) for t in taxa}
            draws=[]
            for sample in samples:
                parts=[groups[t] for t in sample if len(groups[t])]
                if not parts:raise ValueError("empty bootstrap")
                draws.append(float(np.concatenate(parts).mean()))
            ci=np.quantile(draws,[.025,.975])
            metrics[metric]={"mean":float(cells[metric].mean()),"ci95":[float(ci[0]),float(ci[1])]}
        result["designs"][design]={"retained_records":int(eligible.sum()),
           "excluded_records":int((~eligible).sum()),"evaluable_taxon_fold_cells":len(cells),
           "metrics":metrics}
    (ROOT/"c1_interaction_nonfallback_sensitivity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

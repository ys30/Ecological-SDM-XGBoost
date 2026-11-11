"""Descriptive C1 grid4 paired OOF audit. Not a preregistered C1a/C1b contrast.

C1a/b require fixed-parameter controls; this report compares tuned species
versus tuned pooled only, excluding fallback cells for matched AUC.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

def main():
    base=Path("results/v622")
    files={v:base/f"oof_xgb_{v}_tuned_grid4_s2026_b0.csv.gz" for v in ("species","pooled")}
    frames={v:pd.read_csv(p) for v,p in files.items()}
    keys=["plot","Species","outer_fold"]
    a,b=frames["species"],frames["pooled"]
    joined=a.merge(b,on=keys,how="outer",validate="one_to_one",indicator=True,suffixes=("_species","_pooled"))
    if len(joined)!=9336 or not (joined["_merge"]=="both").all():
        raise ValueError("not full matched OOF coverage")
    if not (joined["pres.abs_species"]==joined["pres.abs_pooled"]).all():
        raise ValueError("label mismatch between model arms")
    out=[]
    for (taxon,fold),g in joined.groupby(["Species","outer_fold"],sort=True):
        y=g["pres.abs_species"].to_numpy(int)
        both=(g["fallback_species"].to_numpy(int)==0)&(g["fallback_pooled"].to_numpy(int)==0)
        r={"Species":taxon,"outer_fold":int(fold),"n":len(g),"positives":int(y.sum()),
           "matched_nonfallback_n":int(both.sum()),"matched_nonfallback_positives":int(y[both].sum())}
        for v in ("species","pooled"):
            pred=g[f"pred_{v}"].to_numpy(float)
            r[f"all_logloss_{v}"]=float(log_loss(y,np.clip(pred,1e-6,1-1e-6),labels=[0,1]))
            r[f"all_brier_{v}"]=float(brier_score_loss(y,pred))
            r[f"fallback_n_{v}"]=int(g[f"fallback_{v}"].sum())
            r[f"matched_auc_{v}"]=float(roc_auc_score(y[both],pred[both])) if len(np.unique(y[both]))==2 else np.nan
        r["matched_auc_delta_pooled_minus_species"]=(r["matched_auc_pooled"]-r["matched_auc_species"])
        out.append(r)
    df=pd.DataFrame(out)
    df.to_csv(base/"c1_grid4_tuned_pooled_vs_species_descriptive.csv",index=False)
    summary={"status":"DESCRIPTIVE_ONLY_NOT_C1A_C1B",
      "design":"grid4_s2026_b0","rows_per_model":len(a),
      "taxon_fold_cells":len(df),
      "matched_auc_evaluable_cells":int(df["matched_auc_delta_pooled_minus_species"].notna().sum()),
      "mean_matched_auc_delta_pooled_minus_species":float(df["matched_auc_delta_pooled_minus_species"].mean()),
      "fallback_rows":{v:int(frames[v]["fallback"].sum()) for v in frames},
      "limitations":["No fixed-parameter XGBoost controls, so C1a and C1b remain untested.",
                     "Matched AUC excludes fallback rows and single-class cells.",
                     "Descriptive comparison is not a multiplicity-adjusted confirmatory result."]}
    (base/"c1_grid4_descriptive_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":
    main()

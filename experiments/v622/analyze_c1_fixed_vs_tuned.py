"""C1a/C1b fixed vs tuned paired outer-OOF audit, two primary spatial designs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT=Path("results/v622")
def main():
    cells=[]
    for design in ("grid4_s2026_b0","km15_b0"):
        for variant in ("species","pooled"):
            t=pd.read_csv(ROOT/f"oof_xgb_{variant}_tuned_{design}.csv.gz")
            f=pd.read_csv(ROOT/f"oof_xgb_{variant}_fixed_{design}.csv.gz")
            keys=["plot","Species","outer_fold"]
            j=t.merge(f,on=keys,how="outer",validate="one_to_one",indicator=True,suffixes=("_tuned","_fixed"))
            if len(j)!=9336 or not (j["_merge"]=="both").all():raise ValueError("unmatched OOF")
            if not (j["pres.abs_tuned"]==j["pres.abs_fixed"]).all():raise ValueError("label mismatch")
            if not (j["fallback_tuned"]==j["fallback_fixed"]).all():raise ValueError("fallback policy mismatch")
            for (sp,fold),g in j.groupby(["Species","outer_fold"],sort=True):
                y=g["pres.abs_tuned"].to_numpy(int)
                matched=(g.fallback_tuned.to_numpy(int)==0)&(g.fallback_fixed.to_numpy(int)==0)
                record={"design":design,"variant":variant,"Species":sp,"outer_fold":int(fold),"n":len(g),
                        "matched_n":int(matched.sum()),"positives":int(y.sum())}
                for arm in ("tuned","fixed"):
                    pred=np.clip(g[f"pred_{arm}"].to_numpy(float),1e-6,1-1e-6)
                    record[f"logloss_{arm}"]=float(log_loss(y,pred,labels=[0,1]))
                    record[f"brier_{arm}"]=float(brier_score_loss(y,pred))
                    record[f"auc_{arm}"]=float(roc_auc_score(y[matched],pred[matched])) if len(np.unique(y[matched]))==2 else np.nan
                for metric in ("auc","logloss","brier"):
                    record[f"delta_{metric}_tuned_minus_fixed"]=record[f"{metric}_tuned"]-record[f"{metric}_fixed"]
                cells.append(record)
    df=pd.DataFrame(cells)
    if len(df)!=160:raise ValueError("expected 2 designs x 2 arms x 8 taxa x 5 folds")
    df.to_csv(ROOT/"c1_fixed_vs_tuned_paired_cells.csv",index=False)
    summary={"status":"DESCRIPTIVE_C1A_C1B_PENDING_FORMAL_INFERENCE",
             "delta":"tuned minus fixed; negative logloss/Brier and positive AUC favor tuning",
             "implementation":"fixed is Python reimplementation of historical R XGBoost parameters",
             "results":{}}
    for (d,v),g in df.groupby(["design","variant"]):
        summary["results"][d+"/"+v]={"cells":len(g),"auc_evaluable":int(g.delta_auc_tuned_minus_fixed.notna().sum()),
            "mean_delta_auc":float(g.delta_auc_tuned_minus_fixed.mean()),
            "mean_delta_logloss":float(g.delta_logloss_tuned_minus_fixed.mean()),
            "mean_delta_brier":float(g.delta_brier_tuned_minus_fixed.mean())}
    (ROOT/"c1_fixed_vs_tuned_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()

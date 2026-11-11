"""Grid4 sensitivity: replace Python fixed predictions with observed R hist fixed OOF.

Use identical nonfallback records in all four arms for both baselines.
All comparisons are paired within taxon-fold, 2000 taxon bootstrap draws.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
ROOT=Path("results/v622")
KEYS=["plot","Species","outer_fold"]
def main():
    r=pd.read_csv(ROOT/"c1_r_fixed_hist_oof_grid4.csv")
    if r.duplicated(KEYS+["variant"]).any():raise ValueError("duplicate R predictions")
    arms={}
    for v in ("species","pooled"):
        for t in ("tuned","fixed"):
            x=pd.read_csv(ROOT/f"oof_xgb_{v}_{t}_grid4_s2026_b0.csv.gz")
            if len(x)!=9336 or x.duplicated(KEYS).any():raise ValueError("bad Python OOF")
            arms[v+"_"+t]=x
    m=arms["species_tuned"][KEYS+["pres.abs"]].rename(columns={"pres.abs":"y"})
    for arm,x in arms.items():
        m=m.merge(x[KEYS+["pred","fallback"]].rename(columns={"pred":"p_"+arm,"fallback":"f_"+arm}),on=KEYS,validate="one_to_one")
    if len(m)!=9336:raise ValueError("join changed population")
    for v in ("species","pooled"):
        x=r.loc[r.variant==v,KEYS+["p_r"]].rename(columns={"p_r":"p_"+v+"_r"})
        m=m.merge(x,on=KEYS,how="left",validate="one_to_one")
    valid=np.ones(len(m),dtype=bool)
    for arm in arms:valid &= m["f_"+arm].eq(0).to_numpy()
    for v in ("species","pooled"):valid &= m["p_"+v+"_r"].notna().to_numpy()
    sub=m.loc[valid].copy()
    if len(sub)==0:raise ValueError("no shared nonfallback observations")
    records=[]
    for (sp,fold),g in sub.groupby(["Species","outer_fold"]):
        y=g.y.to_numpy(int)
        rec={"Species":sp,"outer_fold":int(fold),"n":len(g),"positives":int(y.sum())}
        for metric in ("auc","logloss","brier"):
            def score(p):
                p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
                if metric=="auc":return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
                return float(log_loss(y,p,labels=[0,1])) if metric=="logloss" else float(brier_score_loss(y,p))
            for v in ("species","pooled"):
                tuned=score(g["p_"+v+"_tuned"])
                for baseline,field in (("python","p_"+v+"_fixed"),("r_hist","p_"+v+"_r")):
                    rec[f"{v}_{baseline}_{metric}"]=tuned-score(g[field])
        records.append(rec)
    cells=pd.DataFrame(records)
    taxa=np.array(sorted(m.Species.unique()))
    rng=np.random.default_rng(622)
    samples=rng.choice(taxa,size=(2000,len(taxa)),replace=True)
    result={"status":"EXPLORATORY_GRID4_R_HIST_FIXED_BASELINE_SENSITIVITY",
        "definition":"species tuned-minus-fixed minus pooled tuned-minus-fixed; same nonfallback records in all arms",
        "r_baseline":"R xgb.train hist-aligned, not necessarily historical default tree method",
        "retained_records":len(sub),"excluded_records":len(m)-len(sub),
        "taxon_fold_cells":len(cells),"bootstrap":"2000 taxon resamples seed 622",
        "results":{}}
    for metric in ("auc","logloss","brier"):
        result["results"][metric]={}
        for baseline in ("python","r_hist"):
            c=f"interaction_{baseline}_{metric}"
            cells[c]=(cells[f"species_{baseline}_{metric}"]-cells[f"pooled_{baseline}_{metric}"])
            z=cells[["Species",c]].dropna()
            groups={t:z.loc[z.Species==t,c].to_numpy(float) for t in taxa}
            draws=[]
            for sample in samples:
                chunks=[groups[t] for t in sample if len(groups[t])]
                draws.append(float(np.concatenate(chunks).mean()) if chunks else np.nan)
            finite=np.array(draws);finite=finite[np.isfinite(finite)]
            result["results"][metric][baseline]={"mean":float(z[c].mean()),
                "ci95":np.quantile(finite,[.025,.975]).tolist(),
                "evaluable_cells":len(z)}
    cells.to_csv(ROOT/"c1_r_baseline_interaction_paired_cells.csv",index=False)
    (ROOT/"c1_r_baseline_interaction_sensitivity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

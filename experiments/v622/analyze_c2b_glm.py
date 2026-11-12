"""Prespecified C2b paired GLM response-shape comparison from frozen OOF artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
ROOT=Path("results/v622")
def main():
    rng=np.random.default_rng(622)
    summary={"contrast":"C2b","model_a":"glm_quad_rt","model_b":"glm_lin","bootstrap_draws":2000,"bootstrap_seed":622,"resampling_unit":"taxon","designs":{}}
    cells=[]
    for design in ("grid4_s2026_b0","km15_b0"):
        paths=[ROOT/f"oof_{model}_{design}.csv.gz" for model in ("glm_quad_rt","glm_lin")]
        a,b=(pd.read_csv(p) for p in paths)
        keys=["plot","Species","outer_fold"]
        j=a.merge(b,on=keys,how="outer",validate="one_to_one",indicator=True,suffixes=("_quad","_lin"))
        if len(j)!=9336 or not (j._merge=="both").all() or not (j["pres.abs_quad"]==j["pres.abs_lin"]).all():
            raise RuntimeError("INVALID PAIRED C2B OOF")
        taxa=sorted(j.Species.unique())
        if len(taxa)!=8:raise RuntimeError("EXPECTED 8 TAXA")
        for (taxon,fold),g in j.groupby(["Species","outer_fold"]):
            y=g["pres.abs_quad"].to_numpy(int)
            pq=np.clip(g.pred_quad.to_numpy(float),1e-6,1-1e-6)
            pl=np.clip(g.pred_lin.to_numpy(float),1e-6,1-1e-6)
            eligible=(len(np.unique(y))==2 and not g.fallback_quad.astype(bool).any() and not g.fallback_lin.astype(bool).any())
            rec={"design":design,"Species":taxon,"outer_fold":int(fold),"n":len(g),"positives":int(y.sum()),"auc_eligible":bool(eligible),"fallback_rows":int((g.fallback_quad.astype(bool)|g.fallback_lin.astype(bool)).sum())}
            for metric,func in (("auc",roc_auc_score),("ap",average_precision_score),("logloss",lambda y,p:log_loss(y,p,labels=[0,1])),("brier",brier_score_loss)):
                rec["delta_"+metric]=float(func(y,pq)-func(y,pl)) if metric not in ("auc","ap") or eligible else None
            cells.append(rec)
        sub=pd.DataFrame([c for c in cells if c["design"]==design])
        if len(sub)!=40:raise RuntimeError("EXPECTED 40 TAXON-FOLD CELLS")
        samples=rng.choice(taxa,size=(2000,8),replace=True)
        result={"paired_rows":len(j),"cells":len(sub),"fallback_rows":int((j.fallback_quad.astype(bool)|j.fallback_lin.astype(bool)).sum()),"metrics":{}}
        for metric in ("auc","ap","logloss","brier"):
            col="delta_"+metric
            groups={t:sub.loc[sub.Species==t,col].dropna().to_numpy(float) for t in taxa}
            observed=sub[col].dropna().to_numpy(float)
            draws=[]
            for sample in samples:
                selected=[groups[t] for t in sample if len(groups[t])]
                if selected:draws.append(float(np.concatenate(selected).mean()))
            if len(draws)!=2000:raise RuntimeError("BOOTSTRAP EMPTY")
            lo,hi=np.quantile(draws,[.025,.975])
            result["metrics"][metric]={"mean":float(observed.mean()),"ci95":[float(lo),float(hi)],"evaluable_cells":int(len(observed))}
        summary["designs"][design]=result
    aucs=[summary["designs"][d]["metrics"]["auc"] for d in ("grid4_s2026_b0","km15_b0")]
    means=[x["mean"] for x in aucs]
    supported=(means[0]*means[1]>0 and all(abs(m)>=.01 for m in means) and any(x["ci95"][0]>0 or x["ci95"][1]<0 for x in aucs))
    negligible=all(x["ci95"][0]>=-.01 and x["ci95"][1]<=.01 for x in aucs)
    summary["prespecified_auc_decision"]="SUPPORTED" if supported else "NEGLIGIBLE" if negligible else "INCONCLUSIVE"
    summary["limitations"]=["Only eight taxon bootstrap clusters","No multiplicity adjustment","Two spatial designs share the same observations","Python GLM adapter is not verified as historically R-identical"]
    pd.DataFrame(cells).to_csv(ROOT/"c2b_paired_taxon_fold_cells.csv",index=False)
    (ROOT/"c2b_bootstrap_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()

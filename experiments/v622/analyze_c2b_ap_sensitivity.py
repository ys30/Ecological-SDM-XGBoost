"""C2b AP heterogeneity and matched nonfallback proper-score sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,log_loss,brier_score_loss
ROOT=Path("results/v622")
def main():
    all_taxa=[]
    output={"status":"C2B_EXPLORATORY_SENSITIVITY","seed":622,"bootstrap_draws":2000,"designs":{}}
    for design in ("grid4_s2026_b0","km15_b0"):
        a=pd.read_csv(ROOT/f"oof_glm_quad_rt_{design}.csv.gz")
        b=pd.read_csv(ROOT/f"oof_glm_lin_{design}.csv.gz")
        j=a.merge(b,on=["plot","Species","outer_fold"],validate="one_to_one",suffixes=("_quad","_lin"))
        if len(j)!=9336 or not (j["pres.abs_quad"]==j["pres.abs_lin"]).all():raise ValueError("invalid paired OOF")
        rows=[]
        for (taxon,fold),g in j.groupby(["Species","outer_fold"]):
            y=g["pres.abs_quad"].to_numpy(int)
            valid=(np.unique(y).size==2 and not g.fallback_quad.astype(bool).any() and not g.fallback_lin.astype(bool).any())
            non=g.loc[~(g.fallback_quad.astype(bool)|g.fallback_lin.astype(bool))]
            rec={"design":design,"Species":taxon,"outer_fold":int(fold),"n":len(g),"positives":int(y.sum()),"prevalence":float(y.mean()),"ap_eligible":bool(valid),"nonfallback_n":len(non)}
            if valid:
                rec["delta_ap"]=float(average_precision_score(y,g.pred_quad)-average_precision_score(y,g.pred_lin))
            else:rec["delta_ap"]=np.nan
            if len(non):
                yn=non["pres.abs_quad"].to_numpy(int)
                pq=np.clip(non.pred_quad.to_numpy(float),1e-6,1-1e-6)
                pl=np.clip(non.pred_lin.to_numpy(float),1e-6,1-1e-6)
                rec["delta_logloss_nonfallback"]=float(log_loss(yn,pq,labels=[0,1])-log_loss(yn,pl,labels=[0,1]))
                rec["delta_brier_nonfallback"]=float(brier_score_loss(yn,pq)-brier_score_loss(yn,pl))
            rows.append(rec)
        cells=pd.DataFrame(rows)
        if len(cells)!=40:raise ValueError("expected 40 cells")
        taxa=sorted(cells.Species.unique())
        taxa_out=[]
        for taxon,g in cells.groupby("Species"):
            v=g.delta_ap.dropna()
            taxa_out.append({"taxon":taxon,"ap_evaluable_folds":len(v),"mean_delta_ap":float(v.mean()) if len(v) else None,"positive_ap_folds":int((v>0).sum()),"total_positives":int(g.positives.sum()),"prevalence":float(g.positives.sum()/g.n.sum())})
        output["designs"][design]={"ap_evaluable_cells":int(cells.ap_eligible.sum()),"ap_positive_cells":int((cells.delta_ap>0).sum()),"taxa":taxa_out,"nonfallback_rows":int(cells.nonfallback_n.sum()),"metrics":{}}
        rng=np.random.default_rng(622)
        samples=rng.choice(taxa,size=(2000,len(taxa)),replace=True)
        for metric in ("delta_logloss_nonfallback","delta_brier_nonfallback"):
            groups={t:cells.loc[cells.Species==t,metric].dropna().to_numpy(float) for t in taxa}
            draws=[float(np.concatenate([groups[t] for t in sample if len(groups[t])]).mean()) for sample in samples]
            v=cells[metric].dropna().to_numpy(float)
            output["designs"][design]["metrics"][metric]={"mean":float(v.mean()),"ci95":[float(x) for x in np.quantile(draws,[.025,.975])],"cells":len(v)}
        all_taxa.extend(rows)
    pd.DataFrame(all_taxa).to_csv(ROOT/"c2b_ap_heterogeneity_nonfallback_cells.csv",index=False)
    (ROOT/"c2b_ap_heterogeneity_nonfallback.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2),flush=True)
if __name__=="__main__":main()

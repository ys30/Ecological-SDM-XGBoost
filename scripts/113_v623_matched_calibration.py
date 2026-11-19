#!/usr/bin/env python3
"""Matched model calibration diagnostics using identical pooled OOF labels per taxon."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss
BASE=Path("results/v623_matched_three_model")
X=Path("results/v623_fixed_xgb_outer_oof/outer_oof_predictions.csv")
G=Path("results/v623_matched_glm_10km/glm_oof_predictions.csv")
def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert json.loads((BASE/"manifest.json").read_text())["status"]=="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED"
    xr=read(X);gr=read(G)
    assert len(xr)==9336 and len(gr)==18672
    ref={r["id"]:r for r in xr}
    models={"xgboost":ref}
    for variant in ("linear","quadratic"):
        rr=[r for r in gr if r["variant"]==variant]
        models["glm_"+variant]={r["id"]:r for r in rr}
    assert all(len(d)==9336 and set(d)==set(ref) for d in models.values())
    for d in models.values():
        assert all(all(str(r[k])==str(ref[id][k]) for k in ("y","taxon","plot","outer_fold")) for id,r in d.items())
    bins=[];summaries=[]
    taxa=sorted({r["taxon"] for r in xr})
    for taxon in taxa:
        ids=[r["id"] for r in xr if r["taxon"]==taxon]
        y=np.array([int(ref[id]["y"]) for id in ids],dtype=float)
        for name,d in models.items():
            p=np.array([float(d[id]["p"]) for id in ids])
            if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError("Invalid p")
            order=np.argsort(p,kind="stable")
            for j,idx in enumerate(np.array_split(order,10),start=1):
                bins.append(dict(taxon=taxon,model=name,bin=j,n=len(idx),
                    mean_prediction=float(np.mean(p[idx])),observed_fraction=float(np.mean(y[idx])),
                    positives=int(y[idx].sum())))
            eps=1e-12
            z=np.log(np.clip(p,eps,1-eps)/(1-np.clip(p,eps,1-eps)))
            # Calibration-in-the-large: mean prediction minus observed rate, no refit.
            summaries.append(dict(taxon=taxon,model=name,n=len(ids),prevalence=float(y.mean()),
                mean_prediction=float(p.mean()),calibration_in_large=float(p.mean()-y.mean()),
                logloss=float(log_loss(y,p,labels=[0,1])),
                brier=float(brier_score_loss(y,p)),
                logit_min=float(z.min()),logit_max=float(z.max())))
    BASE.mkdir(parents=True,exist_ok=True)
    write(BASE/"calibration_equal_count_deciles.csv",bins)
    write(BASE/"calibration_taxon_summary.csv",summaries)
    # Equal-taxon calibration bias and absolute bias: descriptive summaries, not confidence intervals.
    macro=[]
    for name in models:
        group=[r for r in summaries if r["model"]==name]
        macro.append(dict(model=name,taxa=len(group),
            mean_signed_bias=float(np.mean([r["calibration_in_large"] for r in group])),
            mean_absolute_bias=float(np.mean([abs(r["calibration_in_large"]) for r in group])),
            median_absolute_bias=float(np.median([abs(r["calibration_in_large"]) for r in group])),
            mean_logloss=float(np.mean([r["logloss"] for r in group])),
            mean_brier=float(np.mean([r["brier"] for r in group]))))
    write(BASE/"calibration_macro_summary.csv",macro)
    print(json.dumps({"calibration_macro_summary":macro},indent=2))
    (BASE/"calibration_notes.json").write_text(json.dumps(dict(
        status="DESCRIPTIVE_OOF_CALIBRATION",n_models=3,n_taxa=8,n_deciles_per_taxon=10,
        n_bin_rows=len(bins),n_summary_rows=len(summaries),
        interpretation="Calibration-in-the-large = mean predicted probability minus observed prevalence; 0 is ideal.",
        cautions=["Equal-count deciles are computed separately for each taxon and model.",
                  "Rare taxa with 6 positives can have many zero-event bins; do not infer smooth reliability.",
                  "Do not fit a posthoc calibrator on held-out labels and report its apparent performance.",
                  "Plots are spatially dependent; no row-level independence-based confidence intervals."]),indent=2)+"\n")
    print(json.dumps({"status":"PASS","decile_rows":len(bins),"summary_rows":len(summaries),
                      "taxa":len(taxa),"models":len(models)},indent=2))
if __name__=="__main__":main()

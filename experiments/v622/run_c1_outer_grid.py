"""Frozen-only C1 nested outer predictions for the historical grid4 design.

Refuses to execute until an explicit, reviewed freeze marker is committed.
No outer labels enter inner tuning or final fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.xgb_inner_runner import run_c1_inner_tuning, _features
from experiments.v622.xgb_backend import XGBoostC1Backend
from experiments.v622.xgb_candidate_sampler import generate_candidates

ROOT=Path("experiments/v622")
INPUTS={
 "data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
 "results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze-marker",default="experiments/v622/docs/FROZEN_C1.json")
    parser.add_argument("--output-dir",default="results/v622")
    args=parser.parse_args()
    marker=Path(args.freeze_marker)
    if not marker.is_file():
        raise SystemExit("REFUSED: C1 plan is PRE-FREEZE; reviewed FROZEN_C1.json required")
    lock=json.loads(marker.read_text())
    if lock.get("status")!="FROZEN" or lock.get("design")!="grid4_s2026_b0":
        raise SystemExit("REFUSED: invalid C1 freeze marker")
    if lock.get("inputs")!=INPUTS:
        raise SystemExit("REFUSED: frozen input hashes differ")
    manifest=ROOT/"docs/EMS_v6.22_manifest_sha256.txt"
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if lock.get("manifest_sha256")!=manifest_sha:
        raise SystemExit("REFUSED: freeze marker does not pin current manifest")
    for line in manifest.read_text().splitlines():
        if line.strip():
            expected,relative=line.split("  ",1)
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise SystemExit(f"REFUSED: modified preregistration file {relative}")
    for path,expected in INPUTS.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise SystemExit(f"REFUSED: SHA mismatch {path}")
    data=pd.read_csv("data/train.csv")
    assignments=pd.read_csv("results/spatial_2d_block_assignments.csv")
    if not assignments["plot"].is_unique or set(assignments["plot"])!=set(data["plot"]):
        raise ValueError("invalid plot assignments")
    taxa=sorted(data["Species"].unique())
    candidates=generate_candidates()
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    for variant in ("species","pooled"):
        oof=[]; tuning=[]
        for fold in sorted(assignments["fold"].unique()):
            test_plots=set(assignments.loc[assignments["fold"]==fold,"plot"])
            train=data.loc[~data["plot"].isin(test_plots)].copy()
            test=data.loc[data["plot"].isin(test_plots)].copy()
            if set(train["plot"]) & set(test["plot"]):
                raise ValueError("outer plot leakage")
            inner=spatial_inner_folds(data,test_plots,outer_fold=int(fold))
            mapping=dict(zip(inner["plot"],inner["inner_fold"]))
            backend=XGBoostC1Backend(variant,seed=100+int(fold))
            tuned=run_c1_inner_tuning(train,mapping,backend.primary,backend.retry,
                                      variant=variant,candidates=candidates)
            winner=tuned["selected_candidate"]
            tuning.extend(dict(outer_fold=int(fold),variant=variant,selected=(x["candidate_id"]==winner["candidate_id"]),**x)
                          for x in tuned["candidate_scores"])
            for taxon in taxa:
                tr_taxon=train.loc[train["Species"]==taxon]
                te=test.loc[test["Species"]==taxon]
                fit_train=tr_taxon if variant=="species" else train
                a,b=_features(fit_train,te,variant,np.asarray(taxa,dtype=object))
                labels=fit_train["pres.abs"].to_numpy()
                fallback=False
                error=""
                # Eligibility is evaluated on the fitting population, while
                # fallback prevalence always comes from the target taxon.
                if labels.sum()<5 or np.unique(labels).size<2:
                    fallback=True
                    error="insufficient_fit_support"
                if not fallback:
                    try:
                        pred=backend.primary(winner,a,labels,b,int(fold),str(taxon))
                    except Exception as exc:
                        error=type(exc).__name__
                        try:
                            pred=backend.retry(winner,a,labels,b,int(fold),str(taxon))
                        except Exception as retry_exc:
                            fallback=True
                            error+=";"+type(retry_exc).__name__
                if fallback:
                    y=tr_taxon["pres.abs"].to_numpy()
                    pred=np.full(len(te),(y.sum()+0.5)/(len(y)+1),dtype=float)
                for (_,row),prob in zip(te.iterrows(),pred):
                    oof.append({"plot":row["plot"],"Species":str(taxon),
                                "outer_fold":int(fold),"pres.abs":int(row["pres.abs"]),
                                "pred":float(prob),"fallback":int(fallback),
                                "fit_error":error,"candidate_id":winner["candidate_id"]})
        df=pd.DataFrame(oof)
        if len(df)!=len(data) or df[["plot","Species"]].duplicated().any():
            raise ValueError("OOF coverage/uniqueness failure")
        if not np.isfinite(df["pred"]).all() or not df["pred"].between(0,1).all():
            raise ValueError("invalid probabilities")
        oof_path=outdir/f"oof_xgb_{variant}_tuned_grid4_s2026_b0.csv.gz"
        tuning_path=outdir/f"tuning_xgb_{variant}_tuned_grid4_s2026_b0.csv"
        if oof_path.exists() or tuning_path.exists():
            raise FileExistsError("refusing to overwrite existing formal output")
        df.to_csv(oof_path,index=False,compression="gzip")
        pd.DataFrame(tuning).to_csv(tuning_path,index=False)
        print(f"PASS FROZEN ONLY variant={variant} rows={len(df)} fallback={df['fallback'].sum()}",flush=True)

if __name__=="__main__":
    main()

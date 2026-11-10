"""Real-input engineering smoke: train/validation GLM without outer predictions or metrics.
Uses deterministic plot partitions ONLY for wiring checks, not registered spatial folds.
No artifacts containing observation-level predictions are written.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .design_matrix import prepare_design_fold
from .glm_adapter import run_glm_inner

def main():
    path=Path("data/train.csv")
    if not path.is_file():
        raise FileNotFoundError("repository train.csv not found")
    frame=pd.read_csv(path)
    required={"plot","Species","pres.abs","rainann","soildepth","tempann","topo"}
    if not required<=set(frame):
        raise ValueError(f"missing required fields: {required-set(frame)}")
    plots=sorted(frame["plot"].unique())
    if len(plots)<30:
        raise ValueError("insufficient plots for engineering partitions")
    outer=set(plots[::5])
    inner=set(plots[1::5])
    if outer & inner:
        raise AssertionError("overlapping plot partitions")
    for model in ("glm_lin","glm_quad_rt","glm_quad_env"):
        train,val,_,design=prepare_design_fold(frame,outer,inner,model,standardize=True)
        tr_mask=~frame["plot"].isin(outer|inner)
        va_mask=frame["plot"].isin(inner)
        n_fit=0
        n_fallback=0
        for species in sorted(frame["Species"].unique()):
            tr_sp=frame.loc[tr_mask,"Species"].eq(species).to_numpy()
            va_sp=frame.loc[va_mask,"Species"].eq(species).to_numpy()
            ytr=frame.loc[tr_mask,"pres.abs"].to_numpy()[tr_sp]
            yva=frame.loc[va_mask,"pres.abs"].to_numpy()[va_sp]
            if not len(ytr) or not len(yva):
                raise AssertionError("empty species partition")
            outcome=run_glm_inner(ytr,yva,train.to_numpy()[tr_sp],val.to_numpy()[va_sp])
            if len(outcome.predictions)!=len(yva) or not np.isfinite(outcome.predictions).all():
                raise AssertionError("invalid validation predictions")
            n_fallback+=int(outcome.fallback)
            n_fit+=int(not outcome.fallback)
        print(f"PASS {model}: 8 taxa; successful_fits={n_fit}; fallbacks={n_fallback}; no outer scoring")
if __name__=="__main__":
    main()

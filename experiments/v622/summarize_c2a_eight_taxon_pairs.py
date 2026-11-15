"""Paired per-taxon diagnostic comparison; do not pool across taxa."""
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results/v622/c2a_batch_DIAGNOSTIC_ONLY/grid4_s2026_b0/fold1")
d=pd.read_csv(root/"eight_taxon_metrics_DIAGNOSTIC_ONLY.csv")
if len(d)!=16 or d.duplicated(["taxon","model"]).any():raise RuntimeError("bad metric table")
a=d[d.model=="glmnet_lin"].set_index("taxon").sort_index()
b=d[d.model=="glmnet_quad_rt"].set_index("taxon").sort_index()
if not a.index.equals(b.index) or not np.array_equal(a.positives,b.positives):
    raise RuntimeError("nonpaired taxa")
out=pd.DataFrame({"taxon":a.index,"positives":a.positives.to_numpy(),
    "outer_fallback":(a.outer_fallback.to_numpy()|b.outer_fallback.to_numpy()),
    "auc_linear":a.auc.to_numpy(),"auc_quad":b.auc.to_numpy(),
    "delta_auc":b.auc.to_numpy()-a.auc.to_numpy(),
    "delta_ap":b.ap.to_numpy()-a.ap.to_numpy(),
    "delta_log_loss":b.log_loss.to_numpy()-a.log_loss.to_numpy(),
    "delta_brier":b.brier.to_numpy()-a.brier.to_numpy()})
out["auc_comparable"]=(~out.outer_fallback)&out.positives.gt(0)&out.positives.lt(237)
out["interpretation"]=np.where(out.outer_fallback,"outer_fallback",
    np.where(~out.auc_comparable,"single_class_test","diagnostic_paired"))
out.to_csv(root/"paired_taxon_comparison_DIAGNOSTIC_ONLY.csv",index=False)
valid=out[out.auc_comparable]
print("PASS paired diagnostic summary: comparable taxa",len(valid),
      "AUC wins",int((valid.delta_auc>0).sum()),
      "AUC losses",int((valid.delta_auc<0).sum()),
      "log-loss improvements",int((valid.delta_log_loss<0).sum()),flush=True)
print(out.to_string(index=False),flush=True)
print("DIAGNOSTIC ONLY: one outer fold; no across-taxon inferential claim",flush=True)

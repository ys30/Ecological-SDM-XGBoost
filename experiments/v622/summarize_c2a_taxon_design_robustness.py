"""Taxon-level robustness and design sensitivity for diagnostic C2a results.

Do not interpret fold observations as independent biological replicates.
"""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
d=pd.read_csv(root/"paired_cell_metrics_DIAGNOSTIC_ONLY.csv")
if len(d)!=80 or d.taxon.nunique()!=8 or set(d.design)!={"grid4","km15"}:
    raise RuntimeError("incomplete paired table")
d["valid_auc"]=d.auc_comparable.astype(str).str.lower().eq("true")
rows=[]
for (design,taxon),g in d.groupby(["design","taxon"],sort=True):
    v=g[g.valid_auc]
    rows.append(dict(design=design,taxon=taxon,folds=len(g),valid_auc_folds=len(v),
        fallback_folds=int(g.any_fallback.astype(str).str.lower().eq("true").sum()),
        single_class_folds=int((~g.valid_auc & ~g.any_fallback.astype(str).str.lower().eq("true")).sum()),
        mean_delta_auc=v.delta_auc.mean() if len(v) else np.nan,
        median_delta_auc=v.delta_auc.median() if len(v) else np.nan,
        auc_win_folds=int((v.delta_auc>0).sum()),
        mean_delta_log_loss=g.delta_log_loss.mean(),
        mean_delta_brier=g.delta_brier.mean()))
tax=pd.DataFrame(rows)
tax.to_csv(root/"taxon_design_robustness_DIAGNOSTIC_ONLY.csv",index=False)
wide=tax.pivot(index="taxon",columns="design",values="mean_delta_auc")
wide["grid4_minus_km15"]=wide["grid4"]-wide["km15"]
wide.to_csv(root/"taxon_cross_design_sensitivity_DIAGNOSTIC_ONLY.csv")
print("PASS taxon-design summary",len(tax),"rows, both-design evaluable taxa",
      int(wide[["grid4","km15"]].notna().all(axis=1).sum()),flush=True)
print(tax.to_string(index=False),flush=True)
print("CROSS-DESIGN",flush=True)
print(wide.to_string(),flush=True)
print("DIAGNOSTIC ONLY: descriptive taxon-level robustness; no significance tests",flush=True)

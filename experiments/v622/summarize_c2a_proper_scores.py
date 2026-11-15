"""Diagnostic proper-score comparison with explicit fallback and denominator policies."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
d=pd.read_csv(root/"paired_cell_metrics_DIAGNOSTIC_ONLY.csv")
if len(d)!=80 or d.duplicated(["design","fold","taxon"]).any():
    raise RuntimeError("expected 80 unique paired cells")
d["any_fallback"]=d.any_fallback.astype(str).str.lower().eq("true")
d["test_single_class"]=(d.positives==0)|(d.positives==d.n)
rows=[]
for (design,taxon),g in d.groupby(["design","taxon"]):
    v=g[~g.any_fallback]
    rows.append(dict(design=design,taxon=taxon,total_folds=len(g),
        nonfallback_folds=len(v),fallback_folds=int(g.any_fallback.sum()),
        single_class_nonfallback_folds=int((~g.any_fallback & g.test_single_class).sum()),
        mean_delta_log_loss=v.delta_log_loss.mean() if len(v) else np.nan,
        mean_delta_brier=v.delta_brier.mean() if len(v) else np.nan,
        logloss_win_folds=int((v.delta_log_loss<0).sum()),
        brier_win_folds=int((v.delta_brier<0).sum()),
        mean_delta_log_loss_including_fallback=g.delta_log_loss.mean(),
        mean_delta_brier_including_fallback=g.delta_brier.mean()))
out=pd.DataFrame(rows).sort_values(["design","taxon"])
out.to_csv(root/"proper_score_taxon_design_DIAGNOSTIC_ONLY.csv",index=False)
summary=[]
for design,g in out.groupby("design"):
    for metric in ["log_loss","brier"]:
        values=g["mean_delta_"+metric].dropna()
        summary.append(dict(design=design,metric=metric,taxa=len(values),
            mean_taxon_delta=float(values.mean()) if len(values) else np.nan,
            median_taxon_delta=float(values.median()) if len(values) else np.nan,
            taxa_quad_better=int((values<0).sum()),
            eligible_nonfallback_cells=int(g.nonfallback_folds.sum()),
            single_class_nonfallback_cells=int(g.single_class_nonfallback_folds.sum())))
s=pd.DataFrame(summary)
s.to_csv(root/"proper_score_design_summary_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS proper-score denominator audit",flush=True)
print(s.to_string(index=False),flush=True)
print("DIAGNOSTIC ONLY; negative proper-score delta favors quadratic; single-class folds retained",flush=True)

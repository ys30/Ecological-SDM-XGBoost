"""Compare R fixed hist predictions against frozen Python fixed grid4 OOF."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
root=Path("results/v622")
r=pd.read_csv(root/"c1_r_fixed_hist_oof_grid4.csv")
out={"status":"NUMERICAL_PARITY_DIAGNOSTIC_NOT_ASSERTED_EQUAL","variants":{}}
for variant in ("species","pooled"):
    py=pd.read_csv(root/f"oof_xgb_{variant}_fixed_grid4_s2026_b0.csv.gz")
    a=r[r.variant==variant].drop(columns="variant")
    keys=["plot","Species","outer_fold"]
    m=a.merge(py,on=keys,how="left",validate="one_to_one",indicator=True)
    if not (m._merge=="both").all():raise ValueError("R rows missing Python counterpart")
    if (m.fallback!=0).any():raise ValueError("R rows must be Python nonfallback")
    dif=np.abs(m.p_r-m.pred)
    out["variants"][variant]={"matched_records":len(m),"max_abs_probability_difference":float(dif.max()),
        "mean_abs_probability_difference":float(dif.mean()),
        "fraction_within_1e-6":float(np.mean(dif<1e-6)),
        "fraction_within_1e-3":float(np.mean(dif<1e-3)),
        "pearson_probability_correlation":float(np.corrcoef(m.p_r,m.pred)[0,1])}
(root/"c1_r_python_fixed_hist_parity.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))

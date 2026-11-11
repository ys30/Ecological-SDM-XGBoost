"""Cross-design descriptive synthesis of frozen tuned C1 OOF paired AUC.

Not C1a/C1b: those require v6.21 fixed-parameter controls.
Preserve per-design taxon-fold availability; do not impute missing AUC.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("results/v622")
COL="matched_auc_delta_pooled_minus_species"
DESIGNS={"grid4_s2026_b0":"c1_grid4_tuned_pooled_vs_species_descriptive.csv",
         "km15_b0":"c1_km15_tuned_pooled_vs_species_descriptive.csv"}

def main():
    tables=[]
    for design,name in DESIGNS.items():
        df=pd.read_csv(ROOT/name)
        if len(df)!=40 or df["Species"].nunique()!=8 or df[["Species","outer_fold"]].duplicated().any():
            raise ValueError(f"invalid 8 x 5 taxon-fold table for {design}")
        df["design"]=design
        tables.append(df)
    all_df=pd.concat(tables,ignore_index=True)
    taxa=np.array(sorted(all_df.Species.unique()))
    if len(taxa)!=8: raise ValueError("taxon sets differ")
    by_design={d:all_df[all_df.design==d] for d in DESIGNS}
    groups={d:{t:by_design[d].loc[by_design[d].Species==t,COL].dropna().to_numpy(float) for t in taxa} for d in DESIGNS}
    rng=np.random.default_rng(622)
    draws={d:[] for d in DESIGNS}
    for _ in range(2000):
        sampled=rng.choice(taxa,size=8,replace=True)
        for d in DESIGNS:
            vals=np.concatenate([groups[d][t] for t in sampled if len(groups[d][t])])
            draws[d].append(float(vals.mean()) if len(vals) else np.nan)
    summary={"status":"DESCRIPTIVE_C3D_TWO_DESIGNS_NOT_C1A_C1B",
             "estimand":"unweighted evaluable taxon-fold paired AUC; pooled tuned minus species tuned",
             "bootstrap":{"method":"same resampled taxa in both designs; 2000 draws","seed":622},
             "designs":{},"decision_rule":"support requires same sign, absolute mean >=0.01 in BOTH designs and CI excludes zero in >=1 design",
             "limitations":["Only eight taxon clusters; bootstrap uncertainty is fragile.",
                            "Different spatial designs have different evaluable taxon-fold cells.",
                            "No fixed-parameter C1a/C1b controls and no multiplicity correction."]}
    for d,df in by_design.items():
        vals=df[COL].dropna().to_numpy(float)
        arr=np.array(draws[d])
        if len(vals)==0 or not np.isfinite(arr).all(): raise ValueError("invalid bootstrap coverage")
        summary["designs"][d]={"mean":float(vals.mean()),"ci95":[float(x) for x in np.quantile(arr,[.025,.975])],
            "evaluable_cells":int(len(vals)),"total_cells":40,"fallback_species":int(df.fallback_n_species.sum()),
            "fallback_pooled":int(df.fallback_n_pooled.sum())}
    a,b=list(DESIGNS)
    same_sign=all(summary["designs"][d]["mean"]<0 for d in DESIGNS) or all(summary["designs"][d]["mean"]>0 for d in DESIGNS)
    size=all(abs(summary["designs"][d]["mean"])>=.01 for d in DESIGNS)
    excludes=any((summary["designs"][d]["ci95"][0]>0 or summary["designs"][d]["ci95"][1]<0) for d in DESIGNS)
    summary["supported_by_prespecified_rule"]=bool(same_sign and size and excludes)
    summary["interpretation"]="Same direction, but grid4 magnitude below 0.01 SESOI; not supported by prespecified rule." if same_sign and not size else "Check complete decision criteria."
    (ROOT/"c1_two_design_tuned_descriptive.json").write_text(json.dumps(summary,indent=2)+"\n")
    all_df.to_csv(ROOT/"c1_two_design_tuned_cells.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()

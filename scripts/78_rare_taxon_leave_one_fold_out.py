"""Leave-one-eligible-fold-out sensitivity for rare taxa (descriptive only)."""
from pathlib import Path
import pandas as pd
p=Path("results/alternative_spatial_validation/buffered_four_model")
d=pd.read_csv(p/"paired_taxon_fold_differences.csv")
rows=[]
for (km,v,s,m),g in d[d.species.isin(["nsw19","nsw20","nsw22"])].groupby(["buffer_km","variant","species","metric"]):
    if len(g)<2: continue
    for fold in sorted(g.fold.unique()):
        rest=g[g.fold!=fold]
        rows.append(dict(buffer_km=km,variant=v,species=s,metric=m,omitted_fold=fold,
                         remaining_folds=len(rest),full_mean=g.glm_minus_bayesian.mean(),
                         leave_one_fold_out_mean=rest.glm_minus_bayesian.mean(),
                         sign_flip=(g.glm_minus_bayesian.mean()*rest.glm_minus_bayesian.mean()<0)))
o=pd.DataFrame(rows)
o.to_csv(p/"rare_taxon_leave_one_fold_out.csv",index=False)
s=o.groupby(["buffer_km","variant","species","metric"],as_index=False).agg(
    full_mean=("full_mean","first"),loo_min=("leave_one_fold_out_mean","min"),
    loo_max=("leave_one_fold_out_mean","max"),sign_flips=("sign_flip","sum"),
    n_omissions=("omitted_fold","size"))
s.to_csv(p/"rare_taxon_loo_summary.csv",index=False)
print("QUADRATIC AP LEAVE-ONE-FOLD-OUT")
print(s[(s.variant=="quadratic")&(s.metric=="ap")].to_string(index=False,float_format=lambda x:f"{x:.5f}"))

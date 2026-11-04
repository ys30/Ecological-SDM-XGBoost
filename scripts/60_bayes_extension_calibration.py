"""Quantify Bayesian OOF calibration and taxon heterogeneity on matched predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
files=sorted(Path("downloaded").rglob("*_oof.csv"))
assert len(files)==10
ext=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
base=pd.read_csv("results/bayes_spatial_cv_predictions.csv").rename(columns={"mean_probability":"probability"})
base["variant"]="baseline"
df=pd.concat([base[["id","species","fold","observed","probability","variant"]],ext[["id","species","fold","observed","probability","variant"]]],ignore_index=True)
assert df.groupby("variant").size().eq(9336).all()
out=Path("results/bayes_extensions");out.mkdir(parents=True,exist_ok=True)
rows=[]
for (variant,species),g in df.groupby(["variant","species"]):
    y=g.observed.to_numpy(dtype=float);p=g.probability.to_numpy(dtype=float)
    assert len(g)==1167
    # calibration-in-the-large, without unstable rare-taxon logit regression
    rows.append(dict(variant=variant,species=species,n=len(g),positives=int(y.sum()),
                     prevalence=float(y.mean()),mean_predicted=float(p.mean()),
                     calibration_bias=float(p.mean()-y.mean()),
                     ece_10=float(sum(len(b)/len(g)*abs(b.observed.mean()-b.probability.mean())
                        for _,b in g.groupby(pd.cut(g.probability,np.linspace(0,1,11),include_lowest=True),observed=True))),
                     brier=float(np.mean((y-p)**2))))
r=pd.DataFrame(rows);r.to_csv(out/"taxon_calibration.csv",index=False)
wide=r.pivot(index="species",columns="variant",values=["calibration_bias","ece_10","brier"])
diff=[]
for variant in ["quadratic","no_pool"]:
    for species in wide.index:
        diff.append(dict(variant=variant,species=species,
            delta_calibration_bias=wide.loc[species,("calibration_bias",variant)]-wide.loc[species,("calibration_bias","baseline")],
            delta_ece_10=wide.loc[species,("ece_10",variant)]-wide.loc[species,("ece_10","baseline")],
            delta_brier=wide.loc[species,("brier",variant)]-wide.loc[species,("brier","baseline")]))
d=pd.DataFrame(diff);d.to_csv(out/"taxon_calibration_differences.csv",index=False)
overall=[]
for variant,g in df.groupby("variant"):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    bins=pd.cut(g.probability,np.linspace(0,1,11),include_lowest=True)
    ece=sum(len(b)/len(g)*abs(b.observed.mean()-b.probability.mean()) for _,b in g.groupby(bins,observed=True))
    overall.append(dict(variant=variant,n=len(g),prevalence=float(y.mean()),mean_predicted=float(p.mean()),calibration_bias=float(p.mean()-y.mean()),ece_10=float(ece),brier=float(np.mean((y-p)**2))))
o=pd.DataFrame(overall);o.to_csv(out/"overall_calibration.csv",index=False)
with open(out/"CALIBRATION_REPORT.md","w") as f:
    f.write("# Bayesian extension: OOF calibration and heterogeneity\n\n")
    f.write("Descriptive out-of-fold calibration; ten equal-width bins, pooled ECE. ECE is bin-sensitive and not a proper scoring rule. Comparisons use the archived shorter-chain baseline. No external validation or inferential significance implied.\n\n")
    f.write(o.to_markdown(index=False,floatfmt=".6f"))
    f.write("\n\n## Taxon-level differences\n\n")
    f.write(d.to_markdown(index=False,floatfmt=".6f"))
    f.write("\n")
print(o.to_string(index=False));print(d.to_string(index=False))

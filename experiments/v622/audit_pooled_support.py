"""Audit taxon-specific vs pooled-fit class-support rules on real inner folds.

Diagnostic only. Does not change the preregistered policy, fit models,
read outer-test outcomes, or report model accuracy.
"""
import hashlib
from pathlib import Path
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds

expected = {
    "data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
    "results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
}
for path, sha in expected.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha, f"hash mismatch: {path}"
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/spatial_2d_block_assignments.csv")
assert set(a["plot"])==set(d["plot"]) and a["plot"].is_unique
assert a["fold"].nunique()==5
fold=sorted(a["fold"].unique())[0]
outer=set(a.loc[a["fold"]==fold,"plot"])
train=d.loc[~d["plot"].isin(outer)].copy()
inner=spatial_inner_folds(d,outer,outer_fold=int(fold))
foldmap=dict(zip(inner["plot"],inner["inner_fold"]))
train["inner_fold"]=train["plot"].map(foldmap)
assert train["inner_fold"].notna().all()
taxa=sorted(train["Species"].unique())
records=[]
for f in sorted(train["inner_fold"].unique()):
    tr=train.loc[train["inner_fold"]!=f]
    val=train.loc[train["inner_fold"]==f]
    pooled_pos=int(tr["pres.abs"].sum())
    for taxon in taxa:
        taxon_train=tr.loc[tr["Species"]==taxon]
        taxon_val=val.loc[val["Species"]==taxon]
        assert len(taxon_train)>0 and len(taxon_val)>0
        positives=int(taxon_train["pres.abs"].sum())
        records.append({"inner_fold":int(f),"taxon":str(taxon),
            "taxon_training_positive":positives,
            "pooled_training_positive":pooled_pos,
            "taxon_fallback":positives<5,
            "pooled_fit_supported":pooled_pos>=5 and pooled_pos<len(tr)})
out=pd.DataFrame(records)
assert len(out)==4*len(taxa)
n=int(out["taxon_fallback"].sum())
ambiguous=out.loc[out["taxon_fallback"] & out["pooled_fit_supported"]]
print(f"PASS outer_fold={fold} cells={len(out)} taxon_fallback={n} "
      f"pooled_supported_but_taxon_fallback={len(ambiguous)}",flush=True)
for row in ambiguous.itertuples(index=False):
    print(f"POLICY_AMBIGUITY inner_fold={row.inner_fold} taxon={row.taxon} "
          f"taxon_positives={row.taxon_training_positive} "
          f"pooled_positives={row.pooled_training_positive}",flush=True)
print("DIAGNOSTIC ONLY: no model fits, no policy change, no outer test labels or performance metrics.")

#!/usr/bin/env python3
"""Verify exact fold/taxon training counts and buffer exclusions for XGB and GLMs."""
import csv,json
from pathlib import Path
X=Path("results/v623_fixed_xgb_outer_oof")
G=Path("results/v623_matched_glm_10km")
O=Path("results/v623_matched_three_model")
def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def key(r):return (int(r["outer_fold"]),r["taxon"])
def main():
    x=read(X/"outer_fold_diagnostics.csv")
    g=read(G/"glm_buffer_diagnostics.csv")
    gm=read(G/"glm_fold_metrics.csv")
    xm=read(X/"outer_fold_metrics.csv")
    assert len(x)==40 and len(g)==80 and len(gm)==80 and len(xm)==40
    xx={key(r):r for r in x}
    assert len(xx)==40
    xxm={key(r):r for r in xm}
    gg={(key(r),r["variant"]):r for r in g}
    ggm={(key(r),r["variant"]):r for r in gm}
    assert len(gg)==len(ggm)==80
    output=[]
    for k,ref in sorted(xx.items()):
        for variant in ("linear","quadratic"):
            d=gg[k,variant];m=ggm[k,variant];a=xxm[k]
            checks={
                "n_train":int(ref["train_n"])==int(m["n_train"])==int(a["n_train"]),
                "n_test":int(ref["test_n"])==int(m["n_test"])==int(a["n_test"]),
                "positives_train":int(ref["train_positive"])==int(m["positives_train"])==int(a["positives_train"]),
                "positives_test":int(ref["test_positive"])==int(m["positives_test"])==int(a["positives_test"]),
                "excluded_plots":int(ref["buffered_out_train_plots"])==int(d["n_excluded_train_plots"])
            }
            if not all(checks.values()):raise AssertionError(f"Training mismatch {k} {variant}: {checks}")
            output.append(dict(outer_fold=k[0],taxon=k[1],glm_variant=variant,
                train_n=int(ref["train_n"]),test_n=int(ref["test_n"]),
                positives_train=int(ref["train_positive"]),
                positives_test=int(ref["test_positive"]),
                buffered_out_train_plots=int(ref["buffered_out_train_plots"]),
                status="MATCHED"))
    O.mkdir(parents=True,exist_ok=True)
    with (O/"training_exclusion_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    result=dict(status="PASS",matched_fold_taxon_variant_cells=len(output),
        distinct_fold_taxon_cells=len(xx),compared_variants=["linear","quadratic"],
        checks=["train/test counts","train/test positive counts","excluded training plot counts"],
        limitation="Same hash, shared Haversine implementation and matching counts support equivalence; this audit does not record a cryptographic hash of every actual train ID.")
    (O/"training_exclusion_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

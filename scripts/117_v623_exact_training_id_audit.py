#!/usr/bin/env python3
"""Independently reconstruct exact plot/row ID membership for buffered outer folds.

Checks observed fold/taxon training counts and positives against both model outputs.
No model refit, no raw training IDs exported.
"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from importlib.machinery import SourceFileLoader
xgb=SourceFileLoader("v623_xgb_audit","scripts/102_v623_fixed_xgb_outer_oof.py").load_module()
X=Path("results/v623_fixed_xgb_outer_oof")
G=Path("results/v623_matched_glm_10km")
O=Path("results/v623_matched_three_model")
def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def digest(ids):
    payload="
".join(sorted(map(str,ids)))+"
"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
def main():
    data=read(Path("data/train.csv"))
    folds=read(Path("results/spatial_2d_block_assignments.csv"))
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    assert len(fmap)==len(folds) and set(fmap.values())==set(range(1,6))
    coords={}
    for r in data:
        plot=r["plot"];xy=(float(r["long"]),float(r["lat"]))
        if plot in coords and coords[plot]!=xy:raise ValueError("Coordinate mismatch")
        coords[plot]=xy
    assert set(coords)==set(fmap) and len({r["id"] for r in data})==len(data)
    xdiag={(int(r["outer_fold"]),r["taxon"]):r for r in read(X/"outer_fold_diagnostics.csv")}
    gmetrics={(int(r["outer_fold"]),r["taxon"],r["variant"]):r for r in read(G/"glm_fold_metrics.csv")}
    gdiag={(int(r["outer_fold"]),r["taxon"],r["variant"]):r for r in read(G/"glm_buffer_diagnostics.csv")}
    assert len(xdiag)==40 and len(gmetrics)==len(gdiag)==80
    taxa=sorted({r["Species"] for r in data})
    out=[]
    for fold in range(1,6):
        testplots=sorted((p for p in fmap if fmap[p]==fold),key=int)
        pool=sorted((p for p in fmap if fmap[p]!=fold),key=int)
        dist=xgb.km(np.array([coords[p] for p in pool]),np.array([coords[p] for p in testplots])).min(axis=1)
        keep={p for p,d in zip(pool,dist) if d>10.0}
        assert len(keep)>0
        for taxon in taxa:
            tr=[r for r in data if r["Species"]==taxon and r["plot"] in keep]
            te=[r for r in data if r["Species"]==taxon and r["plot"] in testplots]
            k=(fold,taxon);a=xdiag[k]
            assert len(tr)==int(a["train_n"]) and len(te)==int(a["test_n"])
            assert sum(int(r["pres.abs"]) for r in tr)==int(a["train_positive"])
            assert sum(int(r["pres.abs"]) for r in te)==int(a["test_positive"])
            assert len(pool)-len(keep)==int(a["buffered_out_train_plots"])
            assert digest(r["id"] for r in tr)==a["train_id_sha256"]
            assert digest(r["id"] for r in te)==a["test_id_sha256"]
            for variant in ("linear","quadratic"):
                b=gmetrics[fold,taxon,variant];d=gdiag[fold,taxon,variant]
                assert len(tr)==int(b["n_train"]) and len(te)==int(b["n_test"])
                assert sum(int(r["pres.abs"]) for r in tr)==int(b["positives_train"])
                assert sum(int(r["pres.abs"]) for r in te)==int(b["positives_test"])
                assert len(pool)-len(keep)==int(d["n_excluded_train_plots"])
                assert digest(r["id"] for r in tr)==d["train_id_sha256"]==a["train_id_sha256"]
                assert digest(r["id"] for r in te)==d["test_id_sha256"]==a["test_id_sha256"]
                out.append(dict(outer_fold=fold,taxon=taxon,glm_variant=variant,
                    reconstructed_train_id_sha256=digest(r["id"] for r in tr),
                    reconstructed_test_id_sha256=digest(r["id"] for r in te),
                    reconstructed_keep_plot_sha256=digest(keep),
                    reconstructed_train_n=len(tr),reconstructed_test_n=len(te),status="PASS"))
    O.mkdir(parents=True,exist_ok=True)
    with (O/"reconstructed_training_id_hashes.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    status=dict(status="PASS",cells=len(out),fold_taxon_cells=40,
        interpretation="Independent reconstruction matches fit-time train/test ID SHA-256 digests emitted by both XGBoost and GLM, as well as saved class counts.",
        caveat="Fit-time train/test ID hashes emitted by both model scripts are compared against independently reconstructed membership; this verifies the ID sets, not all preprocessing or feature values.")
    (O/"reconstructed_training_id_audit.json").write_text(json.dumps(status,indent=2)+"
")
    print(json.dumps(status,indent=2))
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Independent checks of the archived 5-fold spatial mapping; not an R RNG replay."""
import csv,json,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
D=Path("data/train.csv")
F=Path("results/spatial_2d_block_assignments.csv")
O=Path("results/v623_matched_three_model")
def load(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows=load(D); folds=load(F)
    assert len(rows)==9336 and len(folds)==1167
    assert len({r["id"] for r in rows})==len(rows)
    plot_rows=defaultdict(list)
    for r in rows:plot_rows[r["plot"]].append(r)
    assert len(plot_rows)==1167
    fm={}
    blocks=defaultdict(set)
    for r in folds:
        p=r["plot"];assert p not in fm
        k=int(r["fold"]);assert 1<=k<=5
        assert r["block"]==r["ix"]+"_"+r["iy"]
        assert 1<=int(r["ix"])<=4 and 1<=int(r["iy"])<=4
        fm[p]=r;blocks[r["block"]].add(k)
        src=plot_rows[p]
        assert len(src)==8
        assert len({v["Species"] for v in src})==8
        for v in src:
            assert abs(float(v["long"])-float(r["long"]))<1e-8
            assert abs(float(v["lat"])-float(r["lat"]))<1e-8
    assert set(fm)==set(plot_rows)
    assert all(len(s)==1 for s in blocks.values())
    assert set(int(v["fold"]) for v in folds)==set(range(1,6))
    # Spatial cell assignment should depend only on coordinates, not species outcomes.
    # Validate that ix/iy are monotone across ordered plot coordinates.
    ordered_lon=sorted(folds,key=lambda r:float(r["long"]))
    ordered_lat=sorted(folds,key=lambda r:float(r["lat"]))
    assert all(int(a["ix"])<=int(b["ix"]) for a,b in zip(ordered_lon,ordered_lon[1:]))
    assert all(int(a["iy"])<=int(b["iy"]) for a,b in zip(ordered_lat,ordered_lat[1:]))
    fold_counts=Counter(int(r["fold"]) for r in folds)
    report=dict(status="PASS_ARCHIVED_MAPPING_INTEGRITY",rows=len(rows),plots=len(fm),
      spatial_blocks=len(blocks),fold_plot_counts=dict(sorted(fold_counts.items())),
      data_sha256=digest(D),fold_sha256=digest(F),
      verified=["plot coordinate equality","one fold per plot","whole spatial blocks in one fold","coordinate grid index monotonicity","complete eight-taxon plot panel"],
      not_verified=["replay of R set.seed(2026) sample ordering","independence of upstream covariate selection from response","source data license","external validation"],
      caveat="This checks archived fold artifact integrity; it is NOT an independent replay of the original R algorithm.")
    O.mkdir(parents=True,exist_ok=True)
    (O/"frozen_spatial_fold_integrity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()

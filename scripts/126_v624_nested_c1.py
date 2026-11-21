#!/usr/bin/env python3
"""Fit the frozen v6.24 C1 contrast and export paired outer predictions."""
import argparse
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from xgboost import XGBClassifier

audit = importlib.import_module("125_v624_buffered_inner_design")
fixed = importlib.import_module("102_v623_fixed_xgb_outer_oof")
grid = importlib.import_module("97_v622_c1_equal_taxon_global_pilot")


def loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))


def predict(config, seed, x, y, train, test):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=seed, n_jobs=1, tree_method="hist", **config)
    model.fit(x[train], y[train])
    p = model.predict_proba(x[test])[:, 1].astype(float)
    if not np.all(np.isfinite(p) & (0 <= p) & (p <= 1)):
        raise ValueError("Invalid model probabilities")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=622)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    source = "data/train.csv"
    folds = "results/spatial_2d_block_assignments.csv"
    rows = fixed.read(source)
    output = Path(f"results/v624_c1/{'pilot' if args.pilot else args.seed}/fold{args.outer}")
    output.mkdir(parents=True, exist_ok=True)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate observation keys")
    taxa = sorted({r["Species"] for r in rows})
    by_taxon = {
        t: sorted((r for r in rows if r["Species"] == t), key=lambda r: int(r["plot"]))
        for t in taxa
    }
    splits = defaultdict(list)
    for record, train, validation, outer_pool, outer_test in audit.design(rows, fixed.read(folds)):
        if record["outer"] == args.outer:
            pool, test = outer_pool, outer_test
            splits[record["taxon"]].append((record, train, validation))
    diagnostics = []
    scores = []
    eligible = []
    candidates = grid.candidates()
    for taxon in taxa:
        taxon_splits = splits[taxon]
        usable = [(r, tr, va) for r, tr, va in taxon_splits if r["usable"]]
        qualifies = (
            taxon_splits[0][0]["outer_train_positive"] >= 5
            and len(usable) >= 2
            and any(r["validation_positive"] > 0 for r, _, _ in usable)
        )
        for record, train, val in taxon_splits:
            ids = by_taxon[taxon]
            diagnostics.append(dict(record, eligible=qualifies,
                                    train_sha256=fixed.ids_sha256(ids[i]["id"] for i in train),
                                    validation_sha256=fixed.ids_sha256(ids[i]["id"] for i in val)))
        if not qualifies:
            continue
        eligible.append(taxon)
        rr = by_taxon[taxon]
        x = np.array([[float(r[k]) for k in fixed.FEATURES] for r in rr])
        y = np.array([int(r["pres.abs"]) for r in rr])
        null = sum(loss(y[val], y[train].mean()) for _, train, val in usable)
        for candidate, config in enumerate(candidates):
            fitted = sum(loss(y[val], predict(config, args.seed, x, y, train, val))
                         for _, train, val in usable)
            scores.append(dict(taxon=taxon, candidate=candidate, total_loss=fitted,
                               null_loss=null, skill=1 - fitted / null))
        print(f"Finished inner fits for {taxon}", flush=True)
    if not eligible or len(scores) != len(eligible) * len(candidates):
        raise ValueError("Incomplete eligible-taxon candidate scores")
    matrix = np.array([r["skill"] for r in scores]).reshape(len(eligible), len(candidates))
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite inner selection skill")
    selected = int(np.argmax(matrix.mean(axis=0)))
    fixed.write(output / "inner_diagnostics.csv", diagnostics)
    fixed.write(output / "candidate_scores.csv", scores)
    predictions = []
    if not args.pilot:
        for taxon in taxa:
            rr = by_taxon[taxon]
            x = np.array([[float(r[k]) for k in fixed.FEATURES] for r in rr])
            y = np.array([int(r["pres.abs"]) for r in rr])
            p_fixed = predict(fixed.PARAMS, args.seed, x, y, pool, test)
            fallback = taxon not in eligible
            p_tuned = p_fixed if fallback or candidates[selected] == fixed.PARAMS else predict(
                candidates[selected], args.seed, x, y, pool, test)
            train_hash = fixed.ids_sha256(rr[i]["id"] for i in pool)
            for i, a, b in zip(test, p_tuned, p_fixed):
                predictions.append(dict(id=rr[i]["id"], plot=rr[i]["plot"], taxon=taxon,
                                        outer=args.outer, seed=args.seed, y=int(y[i]),
                                        tuned=a, fixed=b, fallback=fallback,
                                        train_sha256=train_hash))
        if len({r["id"] for r in predictions}) != len(taxa) * len(test):
            raise ValueError("Incomplete or duplicate outer predictions")
        fixed.write(output / "paired_oof.csv", predictions)
    manifest = dict(outer=args.outer, seed=args.seed, pilot=args.pilot,
                    selector_taxa=eligible, fallback_taxa=sorted(set(taxa) - set(eligible)),
                    selected=selected, parameters=candidates[selected], oof_rows=len(predictions),
                    data_sha256=fixed.sha(source), fold_sha256=fixed.sha(folds),
                    numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    git_sha=os.getenv("GITHUB_SHA"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

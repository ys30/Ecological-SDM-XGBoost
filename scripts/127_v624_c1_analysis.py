#!/usr/bin/env python3
"""Verify paired OOF coverage and summarize seed and geographic sensitivity."""
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

fixed = importlib.import_module("102_v623_fixed_xgb_outer_oof")
SEEDS = (17, 42, 101, 202, 622, 2026)


def point_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def main():
    source = fixed.read("data/train.csv")
    assignments = fixed.read("results/spatial_2d_block_assignments.csv")
    fold_by_plot = {r["plot"]: int(r["fold"]) for r in assignments}
    canonical = {r["id"]: (r["plot"], r["Species"], int(r["pres.abs"]),
                            fold_by_plot[r["plot"]]) for r in source}
    locations = {r["plot"]: (float(r["long"]), float(r["lat"])) for r in source}
    plots = sorted(locations, key=int)
    xy = np.array([locations[p] for p in plots])
    cuts = np.quantile(xy, [.25, .5, .75], axis=0)
    cells = (xy[:, None, :] > cuts).sum(axis=1)
    block_by_plot = dict(zip(plots, 4 * cells[:, 0] + cells[:, 1]))
    output = Path("results/v624_c1_analysis")
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, seed_rows = [], []
    for seed in SEEDS:
        pairs = []
        for fold in range(1, 6):
            folder = Path(f"results/v624_c1/{seed}/fold{fold}")
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["pilot"] or manifest["seed"] != seed or manifest["outer"] != fold:
                raise ValueError(f"Inconsistent fit manifest for {seed}/{fold}")
            rows = fixed.read(folder / "paired_oof.csv")
            if manifest["oof_rows"] != len(rows) or any(
                int(r["seed"]) != seed or int(r["outer"]) != fold for r in rows
            ):
                raise ValueError(f"Wrong forecast scope for {seed}/{fold}")
            pairs.extend(rows)
        keyed = {r["id"]: (r["plot"], r["taxon"], int(r["y"]), int(r["outer"])) for r in pairs}
        if len(pairs) != len(keyed) or keyed != canonical:
            raise ValueError(f"Wrong OOF coverage for seed {seed}")
        y = np.array([int(r["y"]) for r in pairs])
        p = np.array([[float(r["tuned"]), float(r["fixed"])] for r in pairs])
        if not np.all(np.isfinite(p) & (0 <= p) & (p <= 1)):
            raise ValueError(f"Invalid probability for seed {seed}")
        ll = point_loss(y[:, None], p)
        brier = (p - y[:, None]) ** 2
        diff = ll[:, 1] - ll[:, 0]
        blocks = np.array([block_by_plot[r["plot"]] for r in pairs])
        present = np.unique(blocks)
        counts = np.array([(blocks == b).sum() for b in present])
        differences = np.array([diff[blocks == b].sum() for b in present])
        draws = np.random.default_rng(624).integers(len(present), size=(10000, len(present)))
        lo, hi = np.quantile(differences[draws].sum(axis=1) / counts[draws].sum(axis=1),
                             [.025, .975])
        decision = "unresolved"
        if lo > .01:
            decision = "material_benefit"
        elif hi < -.01:
            decision = "material_harm"
        elif lo > -.01 and hi < .01:
            decision = "practical_equivalence"
        seed_rows.append(dict(seed=seed, n=len(y), tuned_logloss=ll[:, 0].mean(),
                              fixed_logloss=ll[:, 1].mean(), benefit=diff.mean(),
                              tuned_brier=brier[:, 0].mean(), fixed_brier=brier[:, 1].mean(),
                              lower=lo, upper=hi, geographic_blocks=len(present),
                              materiality=decision))
        fold_ids = np.array([int(r["outer"]) for r in pairs])
        taxon_ids = np.array([r["taxon"] for r in pairs])
        for taxon in sorted(set(taxon_ids)):
            for fold in range(0, 6):
                mask = (taxon_ids == taxon) & ((fold_ids == fold) | (fold == 0))
                observed = y[mask]
                rank_eligible = len(np.unique(observed)) == 2
                for arm, column in (("tuned", 0), ("fixed", 1)):
                    pred = p[mask, column]
                    metric_rows.append(dict(
                        seed=seed, taxon=taxon, outer=fold, arm=arm, n=len(observed),
                        positives=int(observed.sum()), logloss=ll[mask, column].mean(),
                        brier=brier[mask, column].mean(),
                        auc=roc_auc_score(observed, pred) if rank_eligible else "",
                        ap=average_precision_score(observed, pred) if rank_eligible else "",
                    ))
    fixed.write(output / "seed_spatial_summary.csv", seed_rows)
    fixed.write(output / "taxon_fold_metrics.csv", metric_rows)
    print((output / "seed_spatial_summary.csv").read_text())


if __name__ == "__main__":
    main()

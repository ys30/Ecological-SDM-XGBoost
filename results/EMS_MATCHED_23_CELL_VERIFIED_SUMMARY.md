# EMS preparation: independently checked 23-cell matched comparison

**Source:** `results/matched_bayesian_maxent_xgboost_folds.csv`. These values were independently recalculated from the archived 23-row CSV on 2026-11-03; they are not new model fits or a CI success report.

| Metric (XGBoost minus MaxEnt) | Equal-cell mean | Cells favoring XGBoost | Min cell delta | Max cell delta |
|---|---:|---:|---:|---:|
| ROC AUC | +0.01507868915 | 14/23 | -0.09865648051 | +0.13194614443 |
| Average precision | +0.02658071920 | 13/23 | -0.19498483909 | +0.20687029511 |

## What these numbers do and do not establish

- Both comparisons use the same **23 archived eligible taxon-fold cells**; each cell has equal weight.
- The results do **not** establish a statistically significant superiority of XGBoost, nor separate model algorithm effects from feature set and cross-taxon pooling.
- Spatial folds share training data, and taxa share plot locations; 23 cells must not be treated as 23 independent experiments.
- AP values here reproduce the **archived** definition. `scripts/51_audit_matched_ap_ties.py` audits whether tied prediction scores affect AP and records any differences. Its CI results are not claimed here.
- The pre-existing paired SVG figure axis from -0.26 to +0.26 covers the observed AUC and AP differences above.
- For publication, prioritize common-input comparisons, distance-stratified validation, and matched rare-taxon retrieval before strong claims.

## Associated code

- `scripts/50_matched_distance_stratified_oof.py`: held-out distance stratification (requires CI validation).
- `scripts/51_audit_matched_ap_ties.py`: archived versus threshold-grouped AP (requires CI validation).
- `scripts/52_evaluate_matched_oof.py`: reusable matched evaluation (requires CI validation).
- `scripts/53_build_matched_oof_input.py`: NSW adapter for generic evaluator (requires CI validation).

**Publication policy:** This is a public result audit, not a manuscript draft. Anonymous taxon identifiers are retained.

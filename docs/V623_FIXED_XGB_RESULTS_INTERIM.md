# v6.23 fixed-configuration XGBoost: interim results (2026-11-18)

## Provenance
- Successful CI run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44387473492
- Analysis branch: `research/v6.22-nested-spatial-tuning`.
- Source: `scripts/102_v623_fixed_xgb_outer_oof.py`; integrity check `scripts/103_v623_oof_integrity_gate.py`; summary `scripts/104_v623_performance_report.py`.
- 8 taxa x 5 outer folds = 40 fits, 9,336 OOF predictions, with 10-km exclusion buffer between training and held-out outer plots.
- 30/40 outer taxon-fold tests eligible for AUC/AP; 10 have single-class test labels. All eight pooled taxon prediction sets contain both classes.

## Descriptive results

| Metric | Equal-taxon mean of pooled OOF metric |
|---|---:|
| Log loss | 0.1869051229 |
| Brier score | 0.0559835537 |
| ROC-AUC | 0.7871132590 |
| Average precision | 0.2382871773 |

The pooled ROC-AUC and AP summarize each taxon's OOF predictions across separately fitted outer models; they are **not** averages of outer-fold ROC-AUC/AP. No uncertainty intervals, matched comparisons, or superiority tests have been established.

## Interpretation boundaries
1. Fixed XGBoost parameters (100 trees, depth 2, learning rate 0.03) were predeclared; **no nested hyperparameter optimization** was completed.
2. Relative log loss from inner tuning is not interchangeable with pooled OOF log loss. The mean above is a descriptive pooled-taxon metric.
3. Bayesian/GLM comparisons require verified matching IDs, outer folds, and 10-km exclusion rules. The current XGBoost result alone does not establish a winning model.
4. The original v6.14 manuscript and results remain frozen; this is an independent amendment.
5. The legacy v6.22 pilot workflow may fail separately and should not be confused with the successful v6.23 fixed-configuration workflow.

## Next
- Inspect taxon-level summary artifact and identify rare-taxon calibration and fold heterogeneity.
- Audit comparator OOF row/fold/buffer identity before paired log-loss/Brier differences.
- Build manuscript figures only from verified per-taxon and per-fold results.

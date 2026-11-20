# v6.23 validated protocol freeze — 2026-11-20

**Scope:** matched, within-dataset, five-fold spatially buffered comparison across eight anonymous NSW taxa. **Not** a frozen manuscript, external validation, or nested tuning.

## Frozen analysis contract

- Dataset: 1,167 NSW plots, eight anonymous taxa, 9,336 binary plot–taxon records. Exact NSW `disdat` site/environment/label identity established in archived [source audit #41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532).
- Predictors: `rainann`, `soildepth`, `tempann`, `topo`; do not change features without declaring a new protocol version.
- Outer folds: 4×4 geographic quantile-grid blocks, five folds, deterministic allocation with `set.seed(2026)`; whole plots and blocks stay in one fold. Independent R 4.4.3 replay [#44681382736](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44681382736) successfully matched archived mapping.
- Buffer: 10-km Haversine exclusion from outer held-out plots. Fit-time row membership hashes independently verified by [three-model run #44463218295](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44463218295).
- Models: fixed XGBoost and matched linear/quadratic logistic GLMs, with GLM scaler fitted on each training subset. No hyperparameter search within the v6.23 comparison.
- Evaluation: one outer OOF prediction per plot–taxon record per model, per-taxon metrics averaged equally; log loss/Brier primary and discrimination/calibration complementary. No inferential p-values or universal model-ranking claim.
- Full latest model run [#44678216771](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44678216771); fold-local feature audit [#44525359745](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44525359745).

## Frozen numerical interpretation

| Metric, equal-taxon mean | XGBoost | Linear GLM | Quadratic GLM |
|---|---:|---:|---:|
| Log loss | 0.186905 | 0.204591 | 0.187524 |
| Brier score | 0.055984 | 0.062737 | 0.054642 |
| Absolute calibration-in-the-large bias | 0.002923 | 0.004033 | 0.003887 |

These values come from the previously verified v6.23 matched comparison and are **descriptive**. XGBoost improves mean log loss and Brier relative to linear GLM; compared with quadratic GLM, XGBoost has marginally lower log loss but higher Brier. Do not mix these with historical unmatched Bayesian results.

## Change-control policy

1. Treat the source data, five-fold mapping, 10-km buffer, predictor set, model configurations and primary scoring method as frozen for manuscript v6.23.
2. Correct documentation, citations, figure labels and reproducibility instructions without changing the numerical protocol.
3. Any substantive modification (different training split, tuned hyperparameters, altered labels, new features, different buffer) must be separately versioned and fully rerun; do not silently replace v6.23 figures or claims.
4. Keep historical v6.14 and other results intact.

## Open publication gates (not solved by CI)

- Verify primary-source survey dates, ecological sampling protocol and full bibliography.
- Confirm the exact predictor dictionary and units, especially rainfall; source reuse and redistribution rights.
- Check upstream covariate generation, site selection and any label-dependent preprocessing.
- Finalize complete manuscript text, journal formatting, captions, references, and source/artifact archive.

**Conclusion:** Spatial allocation, matched model membership and fold-local preprocessing are reproducibly checked; the *analytical protocol* is ready to freeze, but the *paper* is not yet submission-ready.

# v6.23 — matched three-model results (manuscript draft)

**Evidence:** [successful primary-fold 10-km matched three-model CI run #44428289030](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44428289030), downloadable comparison artifact ID 14485661518. This is a new exploratory v6.23 experiment, not a revision to frozen v6.14 results.

## Methods
Eight anonymous taxa, 1,167 observations per taxon, 9,336 total OOF rows per model. Five prespecified primary spatial folds. Each outer training set excluded plots within 10 km (Haversine coordinates) of **any** held-out plot. Four predictors (rainann, soildepth, tempann, topo). Fixed XGBoost configuration (100 trees, max depth 2, learning rate 0.03, subsample 0.8, column subsample 0.8, min child weight 1). Logistic GLMs used train-fold-only standardization and fixed L2 penalty C=100; quadratic GLM added squared standardized rainann and tempann. No hyperparameter selection based on these held-out outcomes. The comparator pipeline verified equal data/fold hashes and exact record/label/fold alignment.

## Results: equal-taxon mean paired differences
Differences below are **XGBoost minus comparator**, calculated from each taxon's **pooled OOF** score, then equally averaged across eight taxa. Negative log loss and Brier differences favor XGBoost; positive AUC and AP differences favor XGBoost.

| Metric | vs linear GLM | vs quadratic GLM |
|---|---:|---:|
| Log loss | -0.017686 | -0.000619 |
| Brier | -0.006753 | +0.001341 |
| AUC | -0.013164 | -0.021906 |
| AP | +0.023991 | -0.016299 |

The fixed XGBoost achieved lower mean log loss and Brier than the linear GLM, but slightly lower AUC. Compared with quadratic GLM, log loss was nearly indistinguishable in aggregate and quadratic GLM had higher mean AUC and AP. No model dominated every measure.

## Taxon-level heterogeneity
- XGBoost lower log loss than linear GLM in 5/8 taxa; lower Brier in 7/8.
- XGBoost lower log loss than quadratic GLM in 2/8 taxa; lower Brier in 5/8.
- XGBoost higher AUC than linear GLM in 4/8 and higher AP in 4/8.
- XGBoost higher AUC than quadratic GLM in 3/8 and higher AP in 3/8.

Exploratory exact taxon sign-flip tail fractions for log loss differences were 0.203125 (vs linear) and 0.843750 (vs quadratic); for Brier 0.093750 and 0.812500. These are **not confirmatory p-values**. Spatial dependence and only eight taxa preclude strong population-level inference.

## Interpretation limitations
1. AUC/AP in this comparison are calculated on pooled OOF predictions **within each taxon**, not as averages of outer-fold AUC/AP. Ten of 40 outer taxon-fold tests contain one class, making fold-level AUC/AP ineligible; pooled taxon scores are still defined.
2. The comparison is between **fixed** XGBoost and **fixed** regularized GLMs, not a nested tuned-model tournament. The earlier v6.22 nested-tuning plan was not feasible under its originally prespecified inner-fold design.
3. Taxa are anonymized; external-site/temporal validation has not been demonstrated.
4. The historical Bayesian predictions align by held-out row, but an identical 10-km buffered Bayesian training protocol has not been established. Bayesian must not appear as a controlled third comparator in this new experiment.
5. Plotted taxon score differences are descriptive. Calibration and rare-taxon reliability need separate scrutiny.

## Planned manuscript figures
- Per-taxon pooled OOF log loss, Brier, AUC and AP across all three models.
- Paired XGBoost-minus-GLM probability score differences by taxon, with zero reference.
- Future: calibration reliability and rare-taxon focused panels; plot-level uncertainty assessment.

Generated figure files will be attached to the [matched three-model workflow](https://github.com/ys30/Ecological-SDM-XGBoost/actions/workflows/v623-matched-three-model.yml) after its updated CI succeeds.

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

## Rare-taxon sensitivity — verified

[Successful CI #44430257669](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44430257669) stratified taxa into the four lowest observed prevalence taxa (6–39 detections per taxon) and four higher-prevalence taxa (53–270 detections). The threshold is descriptive and outcome-defined, not a preregistered ecological rarity cutoff. Equal-taxon mean differences (XGBoost minus comparator):

| Comparator | Prevalence stratum | Log loss | Brier | AUC | AP |
|---|---|---:|---:|---:|---:|
| Linear GLM | lower four | -0.002174 | -0.001124 | -0.058402 | +0.005871 |
| Linear GLM | upper four | -0.033197 | -0.012382 | +0.032073 | +0.042110 |
| Quadratic GLM | lower four | -0.011809 | -0.001203 | -0.027724 | +0.003728 |
| Quadratic GLM | upper four | +0.010571 | +0.003885 | -0.016088 | -0.036325 |

The apparent XGBoost improvement over linear GLM in proper probability scores is concentrated in the higher-prevalence group. Against quadratic GLM, log-loss contrast changes direction between strata. With only four taxa per stratum these patterns are hypothesis-generating, not evidence of an ecological effect modification.

## Leave-one-taxon-out influence — verified

[Successful CI #44433165003](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44433165003) calculated 64 seven-taxon contrasts. The ranges below are the **minimum and maximum observed seven-taxon estimates**, not confidence intervals.

| Comparator | Metric | Full 8-taxon difference | Seven-taxon range | Sign reversals (of 8) |
|---|---|---:|---:|---:|
| Linear GLM | Log loss | -0.017686 | [-0.022554, -0.005312] | 0 |
| Linear GLM | Brier | -0.006753 | [-0.008236, -0.002003] | 0 |
| Linear GLM | AUC | -0.013164 | [-0.027507, +0.012681] | 1 |
| Linear GLM | AP | +0.023991 | [+0.010584, +0.033328] | 0 |
| Quadratic GLM | Log loss | -0.000619 | [-0.006134, +0.005858] | 1 |
| Quadratic GLM | Brier | +0.001341 | [-0.000475, +0.001918] | 1 |
| Quadratic GLM | AUC | -0.021906 | [-0.029985, -0.010865] | 0 |
| Quadratic GLM | AP | -0.016299 | [-0.021652, -0.004596] | 0 |

The XGBoost versus linear GLM log loss, Brier and AP contrast signs were unchanged after excluding any single taxon. XGBoost versus quadratic GLM log loss and Brier were sensitive to a single taxon, reinforcing the absence of a robust proper-score winner between those two models. Direction stability does not imply statistical significance.

## Editorial decision for manuscript

**Main text:** matched primary-fold 10-km protocol; eight-taxon pooled OOF contrasts; rare-taxon stratification as a concise descriptive sensitivity; model non-dominance. **Supplement:** all 64 leave-one-taxon-out contrasts, fold-level one-class diagnostics, training exclusions, per-taxon figures, exact sign-flip exploratory tails. **Do not claim:** Bayesian buffered equivalence, tuned-XGBoost superiority, independent-row confidence intervals, external validation, or causal effects of prevalence.

## Spatial outer-fold influence — verified

[Successful CI #44451175135](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44451175135) removed each of five held-out spatial outer folds in turn **from the scored OOF rows**, without refitting models. Differences are XGBoost minus GLM, equally averaged over eight taxon-specific pooled OOF scores after each omission.

| Comparator | Metric | Full eight-taxon delta | Min five fold-deletion delta | Max five fold-deletion delta | Sign reversals |
|---|---|---:|---:|---:|---:|
| Linear GLM | Log loss | -0.017686 | -0.021376 | -0.014430 | 0/5 |
| Linear GLM | Brier | -0.006753 | -0.008726 | -0.005572 | 0/5 |
| Quadratic GLM | Log loss | -0.000619 | -0.003030 | +0.005290 | 1/5 |
| Quadratic GLM | Brier | +0.001341 | +0.000557 | +0.001951 | 0/5 |

The XGBoost improvement over linear GLM in proper probability scores was directionally stable across all five fold deletions. XGBoost versus quadratic GLM log loss was sensitive to a single held-out region; quadratic GLM's lower Brier score remained directionally stable. This is a **descriptive influence analysis**, not a spatial bootstrap, and the five folds must not be treated as independent replicates. No confidence intervals or statistical significance claims follow from these ranges.

## Calibration data availability — verified

[Successful CI #44448864403](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44448864403) generated 240 equal-count calibration decile records and 24 model–taxon summaries for XGBoost, linear GLM, and quadratic GLM. The pipeline computes observed fraction, mean predicted probability, bin positive count, and calibration-in-the-large. The CI success establishes artifact production and identity checks, **not** a finding of satisfactory calibration. Quantitative calibration interpretation requires examining those outputs; avoid stating which model is best calibrated without that audit.

## Quantitative calibration results — verified

[Successful CI #44451628271](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44451628271) computed the following descriptive equal-taxon means using the same spatially held-out predictions and unaltered observed labels:

| Model | Mean signed calibration bias | Mean absolute calibration bias | Median absolute calibration bias | Mean log loss | Mean Brier |
|---|---:|---:|---:|---:|---:|
| XGBoost | +0.001365 | 0.002923 | 0.001180 | 0.186905 | 0.055984 |
| Linear GLM | +0.003877 | 0.004033 | 0.003486 | 0.204591 | 0.062737 |
| Quadratic GLM | +0.003384 | 0.003887 | 0.004270 | 0.187524 | 0.054642 |

Calibration bias is defined **per taxon** as mean predicted probability minus observed prevalence; absolute biases are computed before averaging across taxa. Positive signed values indicate modest mean overprediction. The smallest average absolute bias and log loss occurred with XGBoost; quadratic GLM had the smallest Brier score. These are descriptive comparisons, not inferential significance tests. Calibration-in-the-large does not establish reliability across probability ranges; the 240 model–taxon–decile observations require graphical inspection, especially for taxa with very few detections. The median absolute bias is across eight taxa, not across individual records.

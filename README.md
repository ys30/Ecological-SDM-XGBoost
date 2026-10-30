# Ecological Species Distribution Modeling | R, XGBoost & Spatial Analytics

**Environmental data science portfolio project** · Predicting native tree species presence across forest plots in New South Wales, Australia.

[**Results & technical interpretation**](results/RESULTS_AND_INTERPRETATION.md) · [**Portfolio**](https://ys30.github.io/)

## At a glance

| Dataset | Modeling | Best reported validation result |
|---|---|---|
| 9,336 labeled species–plot records; 8 tree species | Logistic regression → Random forest → XGBoost | **ROC AUC 0.9396 · Log loss 0.1693** |

> **Evidence note:** These are historical **validation** scores documented in the research report, not independently reproduced benchmark results or scores on the unlabeled test set. The dataset has **8.0% positive observations**, making minority-class recall and probabilistic evaluation particularly important.

## Problem & approach

Species occurrence depends on environmental gradients, spatial context, and taxon-specific differences. I built an R-based binary classification workflow to assess whether combining these inputs improves predictions compared with simpler baselines.

- **Data engineering & EDA:** Examine class imbalance, climate and soil distributions, spatial coordinates, categorical features, and data quality.
- **Feature design:** Compare environmental-only, coordinate-only, and combined environmental–geographic–species predictors; encode species and other categorical attributes.
- **Model benchmarking:** Evaluate logistic regression, random forest, and regularized gradient-boosted trees using log loss and ROC AUC.
- **Optimization:** Tune XGBoost with cross-validation, learning rate, tree depth, boosting rounds, and row/column subsampling.
- **Interpretation:** Review feature importance and the limits of spatial inference, then export species-presence probabilities for unlabeled records.

**Stack:** R · tidyverse · caret · randomForest · xgboost · pROC · ggplot2 · ecological & spatial data

## Selected results

### Model comparison — historical validation

| Model | Log loss ↓ | ROC AUC ↑ |
|---|---:|---:|
| Logistic regression | 0.2569 | 0.7882 |
| Random forest | 0.2271 | 0.9343 |
| **XGBoost** | **0.1891** | **0.9369** |

### Feature-set experiments — XGBoost

| Predictors | Log loss ↓ | ROC AUC ↑ |
|---|---:|---:|
| Environmental + species + easting/northing | 0.1737 | 0.9351 |
| **Environmental + species + latitude/longitude** | **0.1693** | **0.9396** |
| Environmental only (no species identity) | 0.3119 | 0.5239 |
| Easting/northing only | 0.3066 | 0.5253 |
| Latitude/longitude only | 0.3053 | 0.5394 |

The combined-feature configuration performed best in the **reported random-split validation**. This comparison does **not** isolate the effect of spatial coordinates, because the environmental-only configuration also omits species identity.

### Classification trade-off

A separate reported tuned-XGBoost classification evaluation found **93.25% accuracy**, **70.65% balanced accuracy**, **43.11% presence recall**, and **98.17% absence specificity**. These numbers highlight why accuracy alone is inadequate with an 8% positive class. The precise configuration linking this confusion matrix to the feature-set results still requires verification.

### What the model learned

The historical gain-based feature-importance plot ranked **annual temperature, latitude, longitude, and annual rainfall** among the leading predictors. Species identity indicators also contributed. These are **predictive associations**, not causal effects or proof of interspecies interactions.

For tables, methodology notes, figure inventory, and limitations, see [Results and interpretation](results/RESULTS_AND_INTERPRETATION.md).

## Portfolio-ready figures

[![Build SDM portfolio figures](https://github.com/ys30/Ecological-SDM-XGBoost/actions/workflows/build-sdm-figures.yml/badge.svg)](https://github.com/ys30/Ecological-SDM-XGBoost/actions/workflows/build-sdm-figures.yml)

The [GitHub Actions workflow](.github/workflows/build-sdm-figures.yml) runs the figure script on changes to its inputs, and can also be started with **Run workflow**. It attempts to commit generated PNGs and CSV summaries back to the repository (requires Actions write permissions). Once the run succeeds, browse [generated figures](results/figures/) and [recomputed species summary](results/species_summary_recomputed.csv).

A standalone [figure-export script](scripts/05_export_portfolio_figures.R) creates five data-derived PNGs (class balance, species counts, environmental histograms, faceted coordinate box plots, and numeric correlation heatmap), plus recalculated species counts and numeric correlations. **Generated files should be considered available only after a successful workflow run; the workflow has been configured, not yet verified as successful.** It avoids treating species identity as observed pairwise co-occurrence.

Run from the repository root:

```bash
Rscript scripts/05_export_portfolio_figures.R
```

After execution, the figures are saved under `results/figures/`. This step requires R and the `ggplot2`, `dplyr`, and `tidyr` packages. Validate the outputs before publishing ecological interpretations.

## Reproducible workflow

```text
data/
  train.csv
  test.csv
scripts/
  00_EDA.R
  00_baselines_logit_rf.R
  01_feature_sets_and_spatial_cv.R
  02_tune_xgboost_caret.R
  03_evaluate_tuned_xgboost.R
  04_train_full_cv_nrounds_predict.R
  05_export_portfolio_figures.R
results/
  RESULTS_AND_INTERPRETATION.md
  submission_xgb_tuned.csv
```

Run the numbered scripts from the repository root with the packages used in each script installed. The code covers exploratory analysis, baseline fitting, feature-set comparisons, tuning, validation, and probability export. **Historical results are not yet accompanied by a complete saved, machine-readable validation benchmark**, so exact numerical reproduction remains a work item.

## Technical judgment & next steps

- **Spatial leakage:** Records can share plots. Current code shows random/stratified validation; plot-grouped and spatially blocked holdouts are needed before claiming transfer to new geographic locations. A script filename mentioning spatial CV is not evidence that blocking was implemented.
- **Rare-presence detection:** Evaluate precision–recall AUC, sensitivity, F1, and calibration alongside ROC AUC and log loss; choose decision thresholds according to survey or conservation costs.
- **Ablation design:** Separate the incremental contributions of species identity, environmental predictors, and coordinates through controlled feature ablations.
- **Reproducibility:** Save validation predictions, folds, seeds, model parameters, metrics, and generated figures; verify temperature scaling and coordinate reference system.
- **Explainability:** Add species-specific partial dependence/SHAP diagnostics, checking correlated-predictor limitations before ecological interpretation.

## Relevance to environmental & climate data science

This case study demonstrates **tabular environmental data analysis, nonlinear ML, model selection, geographic feature engineering, evaluation under class imbalance, and careful communication of uncertainty**—skills transferable to ecological monitoring, climate-risk analytics, remote sensing, and environmental decision support.

[Explore more projects in my portfolio](https://ys30.github.io/)

## License

MIT License

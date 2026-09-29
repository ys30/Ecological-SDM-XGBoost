# Ecological Species Distribution Modeling with XGBoost

An end-to-end **environmental machine-learning workflow** for species presence/absence modeling using environmental and spatial predictors.

## Why this project matters

Species distribution models connect ecological observations with environmental conditions. This project demonstrates a reproducible workflow for moving from exploratory analysis and baseline models to spatial cross-validation, XGBoost tuning, model evaluation, and final prediction.

**Focus:** ecological analytics · environmental predictors · spatial modeling · machine learning  
**Tools:** R · XGBoost · caret · spatial cross-validation

## Analytical workflow

1. **Explore and validate the data** — inspect distributions, missingness, class balance, and candidate predictors.
2. **Establish baselines** — compare logistic-regression and random-forest baselines before tuning a more complex model.
3. **Build feature sets and spatial CV** — organize predictors and use spatially aware validation to reduce overly optimistic performance estimates caused by spatial dependence.
4. **Tune XGBoost** — search model hyperparameters with cross-validation.
5. **Evaluate performance** — assess predictive performance using classification metrics including Accuracy, Precision, F1, AUC, and Log-loss.
6. **Train and predict** — fit the selected workflow and generate final predictions.

## Repository map

```
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
results/
  submission_xgb_tuned.csv
```

## What this repository demonstrates

- Environmental and ecological data preprocessing
- Binary species presence/absence modeling
- Baseline-to-ML model comparison
- Spatially aware cross-validation
- XGBoost hyperparameter tuning
- Multi-metric model evaluation
- Reproducible R analysis organized as a sequential pipeline

## Reproduce the workflow

Run the scripts in numerical order from the repository root. The workflow expects the included training and test data under `data/` and writes model outputs to `results/`.

## Portfolio context

This project is part of my broader work in **environmental and climate data science**, including GIS/remote sensing, environmental exposure analysis, climate-risk modeling, and quantitative decision support.

[View my environmental & climate data portfolio](https://ys30.github.io/)

## License

MIT License

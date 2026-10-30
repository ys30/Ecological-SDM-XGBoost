# Results and interpretation

This page separates **historical reported results** from values independently reproducible from the committed scripts. The repository currently contains prediction output, but not a committed machine-readable model-comparison table or validation log. Research manuscript drafts do provide numerical results and embedded plots. Consequently, historical metrics below are **provisional**, not independently verified against saved evaluation output.

## Data and exploratory results

- Training data: **9,336 rows**, eight species, **1,167 observations per species**.
- Unlabeled test data: **7,264 rows**. No held-out test performance can be calculated without labels.
- Presence prevalence: **0.08001 (8.001%)**; absence prevalence approximately **91.999%**. Accuracy alone is therefore a misleading selection metric.
- Training environmental summaries: annual rainfall mean **1,338** (range **626–2,632**); soil depth mean **1,059** (range **800–1,350**); temperature variable `tempann` mean **154.6** (range **90–194**, verify scaling/units before interpreting in °C); topography mean **5.24** (range **−56–145**). Soil fertility is coded 1–5 and disturbance 1–4; these are categorical codes, not continuous physical measurements.
- Coordinates: longitude **151.2–153.4**, latitude **−32.71 to −28.33**. Projected `easting` contains negative values; verify its coordinate reference system before calling it UTM.

### Species-wise outcomes

| Species | Presences | Observations | Reported correlation with presence |
|---|---:|---:|---:|
| Eucalyptus carnea | 6 | 1,167 | 0.211 |
| Angophora costata | 244 | 1,167 | 0.180 |
| Eucalyptus fastigata | 270 | 1,167 | 0.019 |
| Eucalyptus nova-anglica | 53 | 1,167 | −0.048 |
| Eucalyptus blakelyi | 39 | 1,167 | −0.065 |
| Corymbia intermedia | 18 | 1,167 | −0.090 |
| Corymbia gummifera | 8 | 1,167 | −0.102 |
| Eucalyptus campanulata | 109 | 1,167 | −0.104 |

**Important validation issue:** The correlations above are transcribed from a historical console table and should **not** be used for ecological inference until recalculated. The committed EDA script assigns correlations using an order-dependent vector; additionally, a binary species dummy's association with the target should be checked against the corresponding species prevalence. For example, the very low prevalence of *E. carnea* cannot be interpreted as strong positive co-occurrence on the basis of this table alone. The species column represents the identity of the observation; the dataset does not by itself establish pairwise co-occurrence within a plot.

## Documented validation results (from project research notes)

The following figures are **reported validation results** in the supplied research documents, not newly executed benchmarks. Different tables may correspond to different feature sets and model configurations; do not merge them into one experiment.

### Algorithm comparison

| Algorithm | Validation log loss ↓ | Validation ROC AUC ↑ |
|---|---:|---:|
| Logistic regression | 0.2569 | 0.7882 |
| Random forest | 0.2271 | 0.9343 |
| XGBoost (initial comparison) | **0.1891** | **0.9369** |

### XGBoost feature-set comparison

| Feature set | Validation log loss ↓ | Validation ROC AUC ↑ |
|---|---:|---:|
| Environmental + species identity + easting/northing | 0.1737 | 0.9351 |
| Environmental + species identity + latitude/longitude | **0.1693** | **0.9396** |
| Environmental only | 0.3119 | 0.5239 |
| Easting/northing only | 0.3066 | 0.5253 |
| Latitude/longitude only | 0.3053 | 0.5394 |

**Comparison caveat:** The environmental-only model also excludes species identity, so the difference from the full models does not isolate the contribution of coordinates. These are validation rather than independently held-out test scores.

### Reported classification metrics for tuned XGBoost

| Metric | Reported value |
|---|---:|
| Accuracy | 93.25% |
| Balanced accuracy | 70.65% |
| Sensitivity (presence recall) | 43.11% |
| Specificity (absence recall) | 98.17% |
| Cohen's kappa | 0.4992 |
| McNemar test | p < 0.001 |

These metrics are from the historical manuscript draft. The exact alignment of this confusion-matrix evaluation with the best feature-set experiment remains to be verified. The high specificity combined with moderate sensitivity highlights missed presences despite high overall accuracy. Neither log loss nor these metrics alone demonstrates probability calibration.

### Reported feature importance

The manuscript's gain-based XGBoost feature-importance figure ranks **annual temperature (`tempann`)** first, followed by **latitude**, **longitude**, **annual rainfall (`rainann`)**, and species indicators including *Angophora costata* and *Eucalyptus fastigata*. This is model importance, not a causal effect estimate.

### Additional XGBoost implementation found in the analysis history

A separate XGBoost script encodes `Species` as an integer using the levels of the training data, combines longitude, latitude, disturbance, rainfall, soil depth, soil fertility, temperature, topography, easting, northing and encoded species identity, and trains `binary:logistic` with **eta = 0.03, max_depth = 5, subsample = 0.8, colsample_bytree = 0.8, nrounds = 500**. It exports `submission_xgb_features.csv` and includes code to save `feature_importance.png` and generate a DALEX species partial-dependence profile. No numerical validation metric is established for this separate run. Integer-encoding a nominal species category imposes an arbitrary order in tree splits; one-hot encoding is generally preferable for this analysis. The DALEX example refers to `final_model`, which must be defined consistently before running.

## Model comparison and selected XGBoost

The analysis scripts implement logistic regression and random forest baselines, feature-set comparisons, tuned XGBoost evaluation, and full-data retraining.

| Metric | Tuned XGBoost | Evidence |
|---|---:|---|
| ROC AUC | **0.9396** | Historical report; pending reproducibility check |
| Log loss | **0.1693** | Historical report; pending reproducibility check |
| Sensitivity at threshold 0.5 | **43.11%** | Historical report; pending reproducibility check |
| Specificity at threshold 0.5 | **98.17%** | Historical report; pending reproducibility check |
| Learning rate | **0.01** | Confirmed in final-training script |
| Maximum tree depth | **4** | Confirmed in final-training script |
| Boosting rounds | **978 (historically reported)** | Script selects this dynamically via five-fold CV; exact number not saved |
| Row / column subsampling | **0.8 / 0.8** | Confirmed for final full-data training |

**Do not read the historical AUC/log-loss as test-set performance.** They refer to an earlier validation evaluation, and no persisted validation prediction file is presently identified. An AUC of 0.9396 indicates discrimination, not proof of probability calibration; log loss is a proper scoring rule but calibration should also be assessed directly (reliability curve, Brier score).

### Figures to regenerate from scripts

1. Presence/absence class distribution.
2. Species-wise presence/absence counts.
3. Environmental predictor distributions.
4. Faceted spatial-coordinate box plots.
5. Numeric-predictor correlation matrix.
6. Validation ROC curves and tuned XGBoost feature importance (scripts generate these during evaluation; image files are not necessarily committed).

## Interpretation and methodological cautions

The pronounced imbalance calls for reporting **PR AUC, recall, precision, specificity, F1, ROC AUC and log loss** together. For conservation applications, choosing a threshold should depend on the relative cost of missed presences versus false positives. A threshold of 0.5 is a default, not an ecological optimum.

Environmental and coordinate predictors can capture habitat gradients, but latitude/longitude can also encode location-specific sampling structure. The repository's current partitioning uses a stratified random split; **spatially independent generalization has not been demonstrated**. Multiple species records can share a plot, so grouping by `plot` (and, preferably, geographically blocked cross-validation) is necessary to assess leakage and transfer to new sites. Species identity is a taxon indicator, not automatically a measure of biotic interaction.

The final model uses five-fold CV to select boosting rounds before retraining on all labeled training rows. **The CV call and final fit differ in row/column subsampling (default 1.0 during CV versus 0.8 in the final fit)**; rerun CV with the same parameters to ensure the selected number of rounds corresponds to the deployed configuration.

## Reproducibility priorities

1. Recalculate the species table directly from `group_by(Species)`, with correct per-species prevalence and correctly aligned correlation calculations.
2. Persist a CSV of LR/RF/XGBoost validation metrics, confusion matrices and predictions, and label the split protocol explicitly.
3. Save all figure outputs, the selected boosting round and random seed.
4. Re-evaluate with plot-grouped and spatially blocked folds before claiming geographic transferability.
5. Verify temperature units and the projected coordinate reference system.

This is a transparent research summary: **reported historical results are retained with provenance labels, while unsupported performance or ecological claims are not presented as verified findings.**

# Paired descriptive differences across matched species–folds

Source: `results/matched_bayesian_maxent_xgboost_folds.csv` (23 matched species–fold rows; five species). Each difference is calculated *within the same species and spatial fold*, then averaged over all 23 rows. These are unweighted descriptive summaries, not independent sampling units or confidence intervals.

| Contrast (first minus second) | Mean ΔAUC | AUC folds higher / lower | Mean ΔAP | AP folds higher / lower |
|---|---:|---:|---:|---:|
| XGBoost − Bayesian | +0.04060 | 17 / 6 | +0.05627 | 17 / 6 |
| MaxEnt − Bayesian | +0.02552 | 18 / 5 | +0.02969 | 17 / 6 |
| XGBoost − MaxEnt | +0.01508 | 14 / 9 | +0.02658 | 13 / 10 |

## Species-specific paired differences

| Species | Folds | XGB−Bayes ΔAUC | MaxEnt−Bayes ΔAUC | XGB−MaxEnt ΔAUC | XGB−Bayes ΔAP | MaxEnt−Bayes ΔAP | XGB−MaxEnt ΔAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Angophora costata* | 5 | +0.07337 | +0.03078 | +0.04259 | +0.14994 | +0.05813 | +0.09181 |
| *Eucalyptus blakelyi* | 3 | +0.02614 | +0.04045 | −0.01431 | +0.03938 | +0.03547 | +0.00391 |
| *Eucalyptus campanulata* | 5 | +0.09666 | +0.04782 | +0.04884 | +0.14207 | +0.04684 | +0.09523 |
| *Eucalyptus fastigata* | 5 | −0.01743 | +0.00517 | −0.02261 | −0.06561 | −0.00096 | −0.06465 |
| *Eucalyptus nova-anglica* | 5 | +0.01847 | +0.00934 | +0.00912 | +0.00883 | +0.01130 | −0.00247 |

## Statistical interpretation

The 23 species–fold rows are **not independent replicates**: folds within a species share the same overall data-generating setting and training partitions overlap. A naive t-test or bootstrap treating all 23 rows as independent would understate dependence. The small number of species (five), unequal evaluable folds (three versus five), and fold-level class imbalance also limit the interpretation of uncertainty intervals. No p-values or confidence intervals are claimed here.

For a future inferential analysis, first define whether the estimand concerns these five species or a wider species population; then use a dependence-aware resampling strategy (e.g., paired spatial-block bootstrap with appropriate refitting, or hierarchical modeling of species/fold differences with strong caveats). Without underlying held-out predictions, even a paired observation-level AUC uncertainty analysis cannot be reconstructed faithfully.

These comparisons use the **original shorter-chain Bayesian predictions**, not the long-chain confirmation predictions. The separate long-chain convergence audit does not alter these matched performance numbers.

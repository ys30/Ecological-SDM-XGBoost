# Bayesian rare-species held-out probability scores

Computed from the **existing** `results/bayes_spatial_cv_predictions.csv` (9,336 out-of-fold rows). This is an evaluation of stored predictions, not a new model fit. Each of the three rare species has 1,167 held-out predictions from five spatial folds. Pooled scores aggregate all 1,167 predictions per species; fold scores are also in [the machine-readable CSV](bayesian_rare_species_probability_scores.csv).

| Species | Presences / 1,167 | Observed prevalence | Mean predicted probability | Brier score ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|
| *nsw19* | 8 | 0.00686 | 0.01036 | 0.007294 | 0.039273 |
| *nsw20* | 18 | 0.01542 | 0.01871 | 0.012959 | 0.045149 |
| *nsw22* | 6 | 0.00514 | 0.00407 | 0.005033 | 0.026652 |

**Interpretation:** These proper probability scores are mathematically defined even for zero-presence test folds. Very low prevalence means low Brier/log loss can arise partly from accurate prediction of common absences; they do not demonstrate reliable positive-case detection. The average predicted probability is a coarse calibration-in-the-large diagnostic, not a full calibration curve. No direct MaxEnt/XGBoost comparison or inferential significance is claimed. Scores are based on the **original shorter-chain Bayesian predictions**, not the separate long-chain confirmation fits.

**Formulae:** Brier = mean((observed − probability)^2); log loss = −mean(observed × ln(probability) + (1 − observed) × ln(1 − probability)). Probabilities are clipped at 1e−12 and 1−1e−12 only for numerical stability in log loss.

**Reproducibility:** Source file `results/bayes_spatial_cv_predictions.csv`, filtered to the three named species and grouped by fold or pooled. The derived CSV includes counts, prevalence, mean prediction, Brier and log loss.

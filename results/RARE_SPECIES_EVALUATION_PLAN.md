# Rare-species evaluation: feasibility and safeguards

## Observed held-out event counts

The source-of-truth [fold audit](SPECIES_SPATIAL_FOLD_PRESENCE_AUDIT.md) covers 1,167 plots and five geographic folds (237, 234, 234, 229 and 233 plots). The three species excluded from the matched five-species benchmark have very few observed presences:

| Species | Total positives | Positive counts in folds 1–5 | AUC-evaluable folds | Minimum positive count among evaluable folds |
|---|---:|---|---:|---:|
| *nsw19* | 8 | 4, 1, 3, 0, 0 | 3/5 | 1 |
| *nsw20* | 18 | 9, 0, 9, 0, 0 | 2/5 | 9 |
| *nsw22* | 6 | 0, 1, 0, 0, 5 | 2/5 | 1 |

AUC is mathematically undefined when a test fold contains zero presences. When a fold has one presence, its AUC is defined but highly discrete and unstable. Neither a zero-positive fold nor a one-positive fold supports a robust ranking of competing algorithms. The 23 eligible fold comparisons in the main five-species benchmark should remain separate from these exploratory rare-species cases.

## Recommended secondary evaluation (not yet executed)

1. **Describe prevalence and test support first.** Report per-species counts and the number of evaluable folds before reporting any discrimination metric.
2. **Evaluate calibration and proper scoring rules on all held-out plots** where comparable, out-of-fold probabilities exist. Brier score and Bernoulli log loss remain defined even when a particular held-out fold has no positives; report them alongside fold prevalence. This is not evidence of positive-case retrieval.
3. **Use pooled out-of-fold predictions cautiously.** A pooled ROC AUC across geographic folds is mathematically computable if the pooled species predictions contain both classes. However, the pooled value can reflect between-fold differences in calibration and baseline risk and should not be presented as equivalent to within-fold geographic transfer discrimination.
4. **Report presence retrieval descriptively.** If operational prioritization is the goal, pre-specify a small set of top-k or effort-budget thresholds and report how many observed presences are captured, with explicit denominators. Avoid tuning thresholds on the held-out test data.
5. **Retain the Bayesian hierarchical model as a partial-pooling case study.** A posterior interval describes uncertainty under model assumptions; it does not compensate for absent external validation positives. The original short-chain matched predictions and the long-chain diagnostic confirmations must remain clearly distinguished.
6. **Avoid pseudo-replication.** The plot is the independent sampling unit for species-specific presence data. Species–fold cells and fold-specific predictions are not independent biological replicates. Do not attach naive t-tests, binomial win-rate p-values or conventional iid bootstrap confidence intervals to model contrasts.

## Publication wording

> Three rare species were excluded from the primary matched-model ranking because observed presences were concentrated in only two or three of the five geographic holdout folds, with one or fewer presences in some folds. The resulting absence of positive test cases made fold-level ROC AUC undefined in multiple holdouts, while folds with a single presence yielded highly unstable discrimination estimates. These species remain ecologically important and warrant a separate descriptive evaluation using proper probability scores and explicitly defined presence-retrieval criteria once complete matched out-of-fold predictions are available.

**Status:** This document is an analysis design and interpretation audit, **not** evidence that rare-species predictions or new models have been run. Before calculating Brier/log loss or top-k capture, verify the existence of matched per-record out-of-fold probabilities for each algorithm and species.

## Prediction artifact inventory gate

The existing Bayesian runner (`scripts/15_bayes_hierarchical_spatial.R`) writes `bayes_spatial_cv_predictions_folds_<folds>.csv` (or `_confirm.csv`) containing `id`, `plot`, `species`, `fold`, `observed`, `mean_probability` and posterior interval columns. This establishes **export capability**, not confirmed availability of those files in the repository or historical CI artifacts. The newly committed standard-library inventory `scripts/26_inventory_rare_species_predictions.py` checks prediction-named CSV files and reports whether their schema and rare-species coverage can be identified. It does not score predictions or assert that three algorithms have matched records. **Gate before empirical scoring:** verify per-record held-out probabilities, fold identity, model identity, and identical record coverage across methods. If missing, produce new predictions with the existing spatial split rather than substituting fitted/training probabilities.

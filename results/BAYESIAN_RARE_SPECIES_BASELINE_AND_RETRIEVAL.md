# Rare-species Bayesian predictions: training-fold prevalence baseline and top-k retrieval

The source is `results/bayes_spatial_cv_predictions.csv`, containing out-of-fold Bayesian probabilities from the **original** shorter-chain fits. For each held-out spatial fold, the baseline probability is the observed prevalence for the same species in the other four folds only (training-only, no held-out labels). All scores below pool the 1,167 held-out predictions for each species. This is a descriptive reference baseline, not a newly trained competitor.

| Species | Bayesian Brier ↓ | Train-fold prevalence Brier ↓ | Bayesian log loss ↓ | Train-fold prevalence log loss ↓ |
|---|---:|---:|---:|---:|
| *nsw19* | 0.007294 | **0.006835** | **0.039273** | 0.043137 |
| *nsw20* | **0.012959** | 0.015385 | **0.045149** | 0.086924 |
| *nsw22* | **0.005033** | 0.005154 | **0.026652** | 0.038934 |

## Presence retrieval in pooled out-of-fold rankings

Records are ranked by descending Bayesian out-of-fold probability within species; the top-k capture is the count of observed presences among the first k of 1,167 plots. These are **pooled rankings across geographic folds**, so they do not substitute for per-fold spatial transfer AUC or a prospectively deployed prioritization rule.

| Species | Total presences | Top 10 | Top 25 | Top 50 | Top 100 |
|---|---:|---:|---:|---:|---:|
| *nsw19* | 8 | 0 | 0 | 0 | 1 |
| *nsw20* | 18 | 4 | 7 | 13 | 18 |
| *nsw22* | 6 | 1 | 1 | 2 | 3 |

## Interpretation

- *C. intermedia* exhibits meaningful descriptive enrichment of presences among high-probability plots, but its 18 presences occur in only two spatial folds. The pooled top-k ranking could reflect fold-level shifts, and should not be presented as an independent external-region test.
- *C. gummifera* has very weak high-priority retrieval in these pooled predictions; its Bayesian Brier score is worse than the training-fold prevalence baseline despite lower log loss.
- *E. carnea* has only six presences, so one additional correctly retrieved plot substantially changes top-k recall.
- These descriptive results use no significance tests or confidence intervals. Fold-specific top-k counts and a stratified random-selection reference are now available below; a matched three-algorithm rare-species analysis remains outstanding.

**Scoring definition:** Brier is mean squared probability error. Log loss is the negative mean Bernoulli log likelihood, clipping probabilities only at 1e-12 for numerical stability. The baseline for a held-out fold is calculated as the positive count in the other four folds divided by their plot count; the held-out fold is never used to set its baseline. The top-k values use existing out-of-fold Bayesian probabilities without selecting k to optimize results.

## Spatial-fold-local retrieval with a fixed regional budget

Unlike the pooled ranking above, the fold-local analysis allocates **k plots to each of five geographic folds**. At k=25, the model screens 125/1,167 plots and captures 3/8 *C. gummifera*, 17/18 *C. intermedia*, and 4/6 *E. carnea* presences. The expected captures under simple random selection of 25 plots in each fold are 0.849, 1.911, and 0.643, giving descriptive enrichment factors of 3.53, 8.90, and 6.22 respectively. The random expected value is calculated analytically as Σ_f (25 × presences_f / n_f); folds with zero observed presences still consume their allocated survey budget.

The two strategies **must not be compared as though they had the same survey cost**: pooled Top 25 screens 25 plots total, whereas fold-local Top 25 screens 125. They also encode different spatial-allocation policies. The fold-local findings describe retrospective out-of-fold rankings, not independently validated prospective survey performance.

| Species | Fold-local Top 10 (50 plots total) | Fold-local Top 25 (125 total) | Fold-local Top 50 (250 total) |
|---|---:|---:|---:|
| *C. gummifera* | 1/8 | 3/8 | 6/8 |
| *C. intermedia* | 9/18 | 17/18 | 18/18 |
| *E. carnea* | 2/6 | 4/6 | 6/6 |

**Files:** [per-fold capture and recall](bayesian_rare_species_within_fold_retrieval.csv), [random expectations and enrichment](bayesian_rare_species_random_baseline.csv), [Supplementary Figure S1](figures/24_rare_species_retrieval_publication.svg), [reproducibility audit](RARE_SPECIES_RETRIEVAL_REPRODUCIBILITY_AUDIT.md).

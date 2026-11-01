# Rare-species Bayesian predictions: training-fold prevalence baseline and top-k retrieval

The source is `results/bayes_spatial_cv_predictions.csv`, containing out-of-fold Bayesian probabilities from the **original** shorter-chain fits. For each held-out spatial fold, the baseline probability is the observed prevalence for the same species in the other four folds only (training-only, no held-out labels). All scores below pool the 1,167 held-out predictions for each species. This is a descriptive reference baseline, not a newly trained competitor.

| Species | Bayesian Brier ↓ | Train-fold prevalence Brier ↓ | Bayesian log loss ↓ | Train-fold prevalence log loss ↓ |
|---|---:|---:|---:|---:|
| *Corymbia gummifera* | 0.007294 | **0.006835** | **0.039273** | 0.043137 |
| *Corymbia intermedia* | **0.012959** | 0.015385 | **0.045149** | 0.086924 |
| *Eucalyptus carnea* | **0.005033** | 0.005154 | **0.026652** | 0.038934 |

## Presence retrieval in pooled out-of-fold rankings

Records are ranked by descending Bayesian out-of-fold probability within species; the top-k capture is the count of observed presences among the first k of 1,167 plots. These are **pooled rankings across geographic folds**, so they do not substitute for per-fold spatial transfer AUC or a prospectively deployed prioritization rule.

| Species | Total presences | Top 10 | Top 25 | Top 50 | Top 100 |
|---|---:|---:|---:|---:|---:|
| *Corymbia gummifera* | 8 | 0 | 0 | 0 | 1 |
| *Corymbia intermedia* | 18 | 4 | 7 | 13 | 18 |
| *Eucalyptus carnea* | 6 | 1 | 1 | 2 | 3 |

## Interpretation

- *C. intermedia* exhibits meaningful descriptive enrichment of presences among high-probability plots, but its 18 presences occur in only two spatial folds. The pooled top-k ranking could reflect fold-level shifts, and should not be presented as an independent external-region test.
- *C. gummifera* has very weak high-priority retrieval in these pooled predictions; its Bayesian Brier score is worse than the training-fold prevalence baseline despite lower log loss.
- *E. carnea* has only six presences, so one additional correctly retrieved plot substantially changes top-k recall.
- These descriptive results use no significance tests or confidence intervals. A strong follow-up would report fold-specific top-k counts and compare against a training-only prevalence baseline on the same records.

**Scoring definition:** Brier is mean squared probability error. Log loss is the negative mean Bernoulli log likelihood, clipping probabilities only at 1e-12 for numerical stability. The baseline for a held-out fold is calculated as the positive count in the other four folds divided by their plot count; the held-out fold is never used to set its baseline. The top-k values use existing out-of-fold Bayesian probabilities without selecting k to optimize results.

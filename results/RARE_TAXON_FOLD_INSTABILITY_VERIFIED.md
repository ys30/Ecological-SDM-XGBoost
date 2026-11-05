# Verified rare-taxon fold instability under spatial buffering

**Successful execution:** [GitHub Actions run #42068335398](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42068335398); artifact `rare-taxon-buffered-fold-instability` contains the fold-level paired differences and diagnostic summaries. Source [scripts/77_rare_taxon_fold_instability.py](../scripts/77_rare_taxon_fold_instability.py). The workflow regenerated the original deterministic alternative 15-cluster, five-fold partition before analysis.

## Held-out detections by alternative fold

| Taxon | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| nsw19 | 1 | 2 | 2 | 3 | 0 | 8 |
| nsw20 | 0 | 16 | 2 | 0 | 0 | 18 |
| nsw22 | 0 | 2 | 1 | 3 | 0 | 6 |

nsw20 has 16/18 detections (88.9%) in fold 2, leaving only two folds eligible for AUC/AP. Zero-positive folds cannot yield a meaningful within-fold AP comparison and do not yield ROC AUC.

## Quadratic GLM minus Bayesian paired AP differences at nominal 10 km

| Taxon | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|
| nsw19 | +0.08333 | +0.02924 | +0.30150 | +0.13958 |
| nsw20 | N/A | +0.09301 | +0.38528 | N/A |
| nsw22 | N/A | −0.12067 | −0.06667 | −0.39795 |

## Across-buffer fold ranges (quadratic AP, GLM minus Bayesian)

| Taxon | Buffer | Eligible folds | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| nsw19 | 5 km | 4 | +0.13879 | +0.03117 | +0.30108 |
| nsw19 | 10 km | 4 | +0.13842 | +0.02924 | +0.30150 |
| nsw19 | 20 km | 4 | +0.15655 | +0.03230 | +0.32575 |
| nsw20 | 5 km | 2 | +0.04945 | −0.04924 | +0.14814 |
| nsw20 | 10 km | 2 | +0.23915 | +0.09301 | +0.38528 |
| nsw20 | 20 km | 2 | +0.20762 | +0.02996 | +0.38528 |
| nsw22 | 5 km | 3 | −0.12570 | −0.40210 | +0.09167 |
| nsw22 | 10 km | 3 | −0.19509 | −0.39795 | −0.06667 |
| nsw22 | 20 km | 3 | −0.16263 | −0.39976 | +0.02096 |

**Interpretation:** nsw19 consistently favors GLM on AP across eligible folds at all buffers, but only 1–3 positives occur per fold. nsw20 has marked fold concentration, and nsw22 changes AP sign across folds at 5 and 20 km. Even when all eligible folds share a sign at 10 km, tiny positive counts mean results are exploratory, not robust taxon-specific superiority. These within-dataset spatial folds are not independent external tests. For full taxon heterogeneity and bootstrap uncertainty, see [heterogeneity](BUFFERED_TAXON_HETEROGENEITY_VERIFIED.md) and [bootstrap](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md).

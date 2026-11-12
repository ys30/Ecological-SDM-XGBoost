# C2b AP heterogeneity and nonfallback sensitivity — verified

Source: [successful Actions run 43279582382](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43279582382). This supplements, but does not replace, [C2b primary analysis](C2B_VERIFIED_RESULTS.md). All effects are quadratic rain/temperature GLM minus linear GLM.

## AP heterogeneity

Of 28 eligible taxon–fold cells under grid4, 22 had positive AP differences (78.6%). Under km15, 21 of 31 eligible cells were positive (67.7%). The observed pattern is not confined to ultra-rare taxa: `nsw18` (244/1,167 presences; 20.9%) had mean AP differences +0.2091 (grid4, 5/5 positive folds) and +0.0581 (km15, 4/5 positive); `nsw24` (109/1,167; 9.3%) had +0.1477 and +0.0633 (5/5 positive in each). Conversely, `nsw25` (53/1,167; 4.5%) was +0.0164 and −0.0145; `nsw19` (8/1,167; 0.69%) was +0.0390 and −0.0020. `nsw23` (270/1,167; 23.1%) showed near-zero mean AP effects (+0.0090 and +0.0008).

These are descriptive means over available folds, not independent tests per taxon. Taxa with very few presences contribute few evaluable folds; no multiplicity correction.

## Excluding fallback predictions

| Metric | grid4 mean [95% taxon bootstrap interval] | km15 mean [95% interval] |
|---|---|---|
| Log loss, quadratic minus linear | −0.002826 [−0.049948,+0.046163] | −0.001715 [−0.054965,+0.050617] |
| Brier, quadratic minus linear | −0.007648 [−0.021184,+0.001251] | −0.007131 [−0.021138,+0.001264] |

Nonfallback matched rows: grid4 8,866 (38 cells), km15 8,639 (37 cells). All proper-score intervals include zero. The earlier primary AUC classification remains **INCONCLUSIVE**.

## Manuscript-ready supplementary paragraph

“Exploratory heterogeneity analyses indicated positive quadratic-minus-linear average-precision differences in 22/28 evaluable taxon–fold cells under grid4 and 21/31 under km15. Improvements were not confined to the rarest taxa: taxa nsw18 (20.9% prevalence) and nsw24 (9.3%) had positive mean AP differences in both spatial designs, whereas nsw25 showed opposite signs across designs. Restricting proper scoring rules to matched nonfallback predictions retained directionally negative log-loss and Brier differences, but all corresponding taxon-cluster bootstrap intervals overlapped zero. Thus, response-shape flexibility may improve presence ranking for selected taxa, without robust evidence of an overall AUC or probability-scoring advantage.”

## Limitations

Eight taxon clusters; no multiplicity adjustment; shared underlying data across spatial designs; nonrandom exclusion of fallback and single-class folds; descriptive fold sign counts do not provide a binomial test; Python GLM has not been established numerically equivalent to the original R implementation. AP changes may be influenced by prevalence and ranking ties. Do not present this secondary sensitivity as confirmation of C2b.

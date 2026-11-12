# C2b verified results — quadratic rain/temperature GLM vs linear GLM

**Evidence:** [GitHub Actions 43269143843](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43269143843) (four OOF arms), [paired inference 43273382524](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43273382524), artifact `13970025587`. Two spatial schemes each have 9,336 matched OOF records and 40 taxon–fold cells. The paired AUC/AP evaluations comprise 28 cells for grid4 and 31 for km15; 470 and 697 fallback rows, respectively. The Python statsmodels GLM is not yet verified as numerically equivalent to historical R GLM.

## Results

All differences are **quadratic minus linear**. AUC and average precision (AP) improvements are positive; log loss and Brier improvements are negative. Percentile intervals from 2,000 taxon-cluster bootstrap draws (seed 622).

| Metric | grid4 mean | grid4 95% interval | km15 mean | km15 95% interval |
|---|---:|---|---:|---|
| AUC (prespecified primary) | +0.02404572 | [−0.03649059, +0.06183036] | +0.00835010 | [−0.01545188, +0.03829453] |
| AP (secondary) | +0.07996240 | [+0.02136003, +0.13841769] | +0.02620440 | [+0.00466481, +0.04955671] |
| Log loss (secondary) | −0.00268445 | [−0.04842198, +0.04495960] | −0.00158623 | [−0.05330220, +0.04662451] |
| Brier (secondary) | −0.00726531 | [−0.02060739, +0.00115718] | −0.00659600 | [−0.02001930, +0.00117899] |

**Prespecified AUC decision: INCONCLUSIVE.** Both AUC intervals include zero, and the km15 mean is less than 0.01. These results do not satisfy the support gate. AP shows consistently positive secondary descriptive intervals, without multiplicity adjustment. Neither proper scoring metric has a 95% interval excluding zero.

## Manuscript-ready Results paragraph

“In the prespecified C2b comparison of quadratic rainfall/temperature and linear logistic response specifications, mean paired ROC AUC differences (quadratic minus linear) were +0.0240 (95% taxon-cluster bootstrap interval −0.0365 to +0.0618) under grid4 spatial holdouts and +0.0084 (−0.0155 to +0.0383) under km15 holdouts. The prespecified AUC support criterion was not met. In secondary analyses, average precision differences were +0.0800 (+0.0214 to +0.1384) and +0.0262 (+0.0047 to +0.0496), respectively, whereas the log-loss and Brier intervals included zero in both schemes. The AP pattern suggests a possible benefit for retrieving presences under spatial holdouts, but should be considered secondary and exploratory rather than confirmatory.”

## Limitations and next checks

Only eight taxon bootstrap clusters; two spatial partitions of the same records; no multiplicity adjustment; AUC/AP excluded paired fallback and single-class cells, while log loss/Brier used all taxon–fold cells; Python GLM implementation not numerically verified against R. Before final publication, inspect taxon-level AP heterogeneity, positive prevalence per fold, and sensitivity to excluding fallback predictions for proper scores. Do not claim external validation or causal effect of adding quadratic terms.

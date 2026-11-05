# Verified rare-taxon fixed false-positive budget ranking diagnostic

Successful [CI #42108171425](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42108171425), artifact `rare-taxon-fp-budget-ranking` ID 13451389216. Script [82_rare_taxon_fp_budget_ranking.py](../scripts/82_rare_taxon_fp_budget_ranking.py). Quadratic GLM and Bayesian buffered OOF predictions are matched by record ID and checked for equal labels. Outputs are aggregate-only. All three species evaluated at 5, 10 and 20 km; budgets 0, 5, 10, 20, 50, 100.

## 10 km buffered: detected positives at equal oracle false-positive budgets

| Taxon | Positives | Model | 0 FP | 5 FP | 10 FP | 20 FP | 50 FP | 100 FP |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| nsw19 | 8 | GLM | 0 | 0 | 1 | 2 | 6 | 7 |
| nsw19 | 8 | Bayesian | 0 | 0 | 0 | 0 | 2 | 4 |
| nsw20 | 18 | GLM | 0 | 4 | 6 | 10 | 12 | 17 |
| nsw20 | 18 | Bayesian | 0 | 1 | 1 | 1 | 10 | 12 |
| nsw22 | 6 | GLM | 0 | 0 | 0 | 0 | 1 | 4 |
| nsw22 | 6 | Bayesian | 1 | 1 | 1 | 2 | 4 | 5 |

At the sampled 10-km budgets, GLM ranks positives earlier for nsw19 and nsw20, while Bayesian ranks positives earlier for nsw22. All boundaries shown have score tie count 1, though ties elsewhere are not ruled out.

**Critical limitation:** These are *oracle* false-positive budgets: the OOF test labels determine the cutoff for each model, and thus the threshold selection uses test outcomes. This is a descriptive ranking diagnostic, **not** a deployable or unbiased threshold selection protocol. It must not be framed as prospective field monitoring performance. Small taxon-specific positive counts (6, 8, 18), within-dataset spatial OOF, and earlier broad bootstrap intervals preclude claims of significant universal model advantage. A real decision protocol needs thresholds fixed from separate validation data and evaluated on untouched test folds.

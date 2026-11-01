# Sensitivity to species versus species–fold weighting

The 23 matched species–fold comparisons span five species, with four species contributing five evaluable folds and *nsw21* contributing three. Thus, a mean across 23 fold rows gives more weight to species with five eligible folds. To examine this design choice, this audit compares (1) equal weight per eligible species–fold and (2) equal weight per species, calculated from species-level mean metrics.

| Contrast | 23-fold mean ΔAUC | 5-species mean ΔAUC | 23-fold mean ΔAP | 5-species mean ΔAP | Species with positive ΔAUC | Species with positive ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost − Bayesian | +0.04060 | +0.03944 | +0.05627 | +0.05492 | 4/5 | 4/5 |
| MaxEnt − Bayesian | +0.02552 | +0.02671 | +0.02969 | +0.03015 | 5/5 | 4/5 |
| XGBoost − MaxEnt | +0.01508 | +0.01273 | +0.02658 | +0.02477 | 3/5 | 3/5 |

All three aggregate pairwise contrasts retain their sign under the alternative weighting. The XGBoost–MaxEnt AUC contrast decreases from +0.01508 to +0.01273, so the magnitude is somewhat sensitive to weighting even though its sign is not.

**Interpretation:** This is a deterministic reweighting sensitivity analysis, not a bootstrap, hypothesis test or confidence interval. Five species cannot support a robust population-level species inference without additional assumptions. The paired comparison uses original shorter-chain Bayesian predictions, while long-chain confirmation assessed separate fold-level fit stability.

**Data:** `results/matched_bayesian_maxent_xgboost_folds.csv` and `results/matched_bayesian_maxent_xgboost_species.csv`. Differences are first-model minus second-model.

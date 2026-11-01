# Bayesian parameter-level convergence audit

Source: the five published `results/bayes_spatial_fold_{1..5}_diagnostics.csv` files. Total: **175 parameter-fold rows** (35 per fold). Thresholds are screening criteria: R-hat > 1.01 or bulk ESS < 400.

| Fold | Parameter | R-hat | Bulk ESS | Flag |
|---|---|---:|---:|---|
| 1 | lp__ | 1.00341 | 319.71 | ESS |
| 2 | lp__ | 1.00738 | 328.91 | ESS |
| 2 | sd_species__z_rainann | 1.01453 | 350.83 | R-hat + ESS |
| 2 | r_species[nsw22,z_rainann] | 1.01117 | 492.64 | R-hat |
| 3 | lp__ | 1.00612 | 190.99 | ESS |
| 3 | sd_species__z_rainann | 1.00324 | 337.76 | ESS |
| 4 | lp__ | 1.00572 | 266.84 | ESS |
| 4 | r_species[nsw23,Intercept] | 0.99952 | 391.87 | ESS |
| 5 | lp__ | 1.00269 | 340.80 | ESS |
| 5 | b_z_tempann | 1.00942 | 351.92 | ESS |
| 5 | r_species[nsw24,z_tempann] | 1.00989 | 362.22 | ESS |
| 5 | r_species[nsw23,z_tempann] | 1.00751 | 369.03 | ESS |
| 5 | r_species[nsw18,z_tempann] | 1.00807 | 373.34 | ESS |
| 5 | b_z_topo | 1.01110 | 1075.71 | R-hat |

**Counts:** 14 unique flagged rows; 3 rows have R-hat > 1.01; 12 rows have bulk ESS < 400. One row violates both criteria.

**Sampler diagnostics:** The ten chains have zero divergences, zero treedepth-limit hits, and minimum BFMI 0.6794. This does not resolve the parameter-level concerns above.

**Next:** prioritize longer runs for folds 2, 3 and 5; also monitor fold 1 and 4 low `lp__` ESS. Export and inspect tail ESS for all parameters before any full-convergence claim. This audit is descriptive; thresholds are conventional screening rules, not hypothesis tests.

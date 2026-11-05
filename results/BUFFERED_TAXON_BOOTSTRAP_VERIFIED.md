# Verified buffered GLM–Bayesian taxon-cluster bootstrap

Source: [successful GitHub Actions run #41930995688](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41930995688), artifact `buffered-paired-taxon-bootstrap`. Paired OOF audit: [#41917257355](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41917257355). Reproducible script: [scripts/74_buffered_taxon_cluster_bootstrap.py](../scripts/74_buffered_taxon_cluster_bootstrap.py).

Twenty thousand bootstrap resamples of **8 anonymized taxa**. For each metric and radius, first average matched GLM-minus-Bayesian differences over eligible folds within taxon, then resample taxa with replacement and average equally across taxa. AUC/AP: 34 eligible taxon×fold cells; Brier/log loss: 40. The intervals are descriptive percentile intervals, not confirmatory tests.

| Radius | Specification | Metric | Equal-taxon mean GLM − Bayes | 95% percentile interval |
|---:|---|---|---:|---|
| 5 km | quadratic | AUC | +0.006769 | [−0.027087, +0.040848] |
| 5 km | quadratic | AP | +0.005738 | [−0.042304, +0.054015] |
| 5 km | quadratic | Brier | +0.000399 | [−0.001493, +0.001733] |
| 5 km | quadratic | Log loss | +0.008518 | [−0.000843, +0.019478] |
| 10 km | quadratic | AUC | +0.006734 | [−0.030186, +0.041214] |
| 10 km | quadratic | AP | +0.021956 | [−0.058637, +0.104825] |
| 10 km | quadratic | Brier | +0.000537 | [−0.001956, +0.002322] |
| 10 km | quadratic | Log loss | +0.009524 | [−0.001215, +0.020920] |
| 20 km | quadratic | AUC | +0.028076 | [−0.027916, +0.095023] |
| 20 km | quadratic | AP | +0.032476 | [−0.042604, +0.106642] |
| 20 km | quadratic | Brier | +0.000682 | [−0.000834, +0.002002] |
| 20 km | quadratic | Log loss | +0.011177 | [−0.010873, +0.037067] |

All linear-specification intervals also include zero; see the run artifact for the full 24-row table. All 24 intervals include zero. Higher AUC/AP is preferable; lower Brier/log loss is preferable. Differences between **pooled** metrics and **equal-taxon mean paired** metrics are expected because these estimands weight taxa and folds differently. Do not conflate the two.

**Limitations:** only eight taxon clusters; intervals can be unstable. All spatial folds come from the same dataset, not independent external validation. Increasing buffer radii also reduces training sample size. Projected coordinate CRS and meter units remain to be independently confirmed. No claim of statistically significant model superiority.

# Four-model taxon-cluster bootstrap — verified

[Successful GitHub Actions run 41872802258](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41872802258) performed 20,000 seeded bootstrap resamples over eight anonymous taxa, taking within-taxon means over available spatial folds before resampling. These are exploratory equal-taxon percentile intervals; eight taxa and shared geographic sampling limit inference. Do not interpret interval crossing zero as a formal test of no effect.

| Model A − Model B | Metric | Equal-taxon mean | 95% bootstrap interval | Taxa favoring A / 8 |
|---|---|---:|---|---:|
| GLM quadratic − Bayesian quadratic | AUC | +0.006439 | [−0.026749, +0.039742] | 5 |
| GLM quadratic − Bayesian quadratic | AP | +0.027660 | [−0.041827, +0.121808] | 4 |
| GLM quadratic − Bayesian quadratic | Brier | +0.000313 | [−0.001643, +0.001686] | 2 |
| GLM quadratic − Bayesian quadratic | Logloss | +0.008431 | [−0.001182, +0.020494] | 2 |
| GLM quadratic − GLM linear | AUC | +0.006719 | [−0.013840, +0.029043] | 4 |
| GLM quadratic − GLM linear | AP | +0.000883 | [−0.073008, +0.051929] | 6 |
| Bayesian quadratic − Bayesian linear | AUC | +0.013088 | [−0.003065, +0.031265] | 6 |
| Bayesian quadratic − Bayesian linear | AP | +0.009112 | [−0.054618, +0.057681] | 6 |
| Bayesian quadratic − Bayesian linear | Brier | −0.006479 | [−0.017946, +0.000453] | 5 |
| Bayesian quadratic − Bayesian linear | Logloss | −0.017405 | [−0.051558, +0.005412] | 5 |

**Interpretation:** All reported intervals span zero. Pooled OOF results favor GLM quadratic on AUC/AP and Bayesian quadratic on Brier/logloss, but neither is a stable universal winner across taxa. Per-taxon quadratic effects are heterogeneous. The equal-taxon AP contrast GLM quadratic versus GLM linear (+0.000883) is much smaller than the equal-cell contrast (+0.016246), underscoring estimand sensitivity. See [matched four-model metrics](FOUR_MODEL_ALTERNATIVE_SPATIAL_VERIFIED.md). No external validation or confirmatory significance is claimed.

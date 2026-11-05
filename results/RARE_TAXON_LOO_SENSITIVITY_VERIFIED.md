# Verified leave-one-eligible-fold-out sensitivity for rare taxa

[Successful GitHub Actions run #42073233897](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42073233897). The `rare-taxon-loo-sensitivity` artifact contains `rare_taxon_leave_one_fold_out.csv` and `rare_taxon_loo_summary.csv`. Reproducible code: [scripts/78_rare_taxon_leave_one_fold_out.py](../scripts/78_rare_taxon_leave_one_fold_out.py).

The analysis averages **fold-level paired AP differences** (quadratic GLM minus quadratic Bayesian), then drops one eligible fold at a time. Positive values favor GLM, negative values favor Bayesian. This is a diagnostic on existing held-out predictions, not refitting or independent validation.

| Taxon | Buffer | Full mean ΔAP | LOO min | LOO max | Sign reversals / omissions |
|---|---:|---:|---:|---:|---:|
| nsw19 | 5 km | +0.13879 | +0.08470 | +0.17467 | 0/4 |
| nsw19 | 10 km | +0.13842 | +0.08405 | +0.17481 | 0/4 |
| nsw19 | 20 km | +0.15655 | +0.10015 | +0.19797 | 0/4 |
| nsw20 | 5 km | +0.04945 | −0.04924 | +0.14814 | 1/2 |
| nsw20 | 10 km | +0.23915 | +0.09301 | +0.38528 | 0/2 |
| nsw20 | 20 km | +0.20762 | +0.02996 | +0.38528 | 0/2 |
| nsw22 | 5 km | −0.12570 | −0.23438 | +0.01250 | 1/3 |
| nsw22 | 10 km | −0.19509 | −0.25931 | −0.09367 | 0/3 |
| nsw22 | 20 km | −0.16263 | −0.25442 | −0.04407 | 0/3 |

nsw19 retains positive AP contrasts under all leave-one-fold-out omissions, whereas nsw20 and nsw22 show reversals at 5 km. However, [held-out detections are sparse and geographically concentrated](RARE_TAXON_FOLD_INSTABILITY_VERIFIED.md): nsw19 has only 8 detections across 4 eligible folds; nsw20 has 16 of 18 detections in fold 2; nsw22 has only 6 detections. Consequently, this direction stability is descriptive and does not establish statistical significance or generalizable superiority. Existing [eight-taxon bootstrap intervals](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md) cross zero.

**Manuscript interpretation:** State that rare-taxon AP model contrasts are sensitive to both spatial buffer and held-out fold composition, with heterogeneous patterns across taxa; do not claim universal benefit from Bayesian partial pooling or GLM.

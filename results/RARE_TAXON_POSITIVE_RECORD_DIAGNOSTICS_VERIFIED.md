# Verified rare-taxon positive-record probability and ranking diagnostics

**Successful GitHub Actions:** [#42078878397](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42078878397). Reproducible script: [79_rare_positive_record_diagnostics.py](../scripts/79_rare_positive_record_diagnostics.py). CI artifact `rare-positive-record-aggregate-diagnostics` contains only fold-level aggregate CSVs, not individual prediction rows. GLM and Bayesian OOF rows are joined by record ID, with matched labels and species checked.

## 10 km buffered quadratic model comparisons

Positive-record average probability and mean rank (smaller rank is better) are given for folds with detections.

| Taxon | Fold | Positives | GLM positive mean p | Bayesian positive mean p | GLM positive mean rank | Bayesian positive mean rank |
|---|---:|---:|---:|---:|---:|---:|
| nsw19 | 1 | 1 | .0535 | .0205 | 4.0 | 6.0 |
| nsw19 | 2 | 2 | .0190 | .0231 | 29.0 | 51.5 |
| nsw19 | 3 | 2 | .0497 | .0169 | 7.0 | 46.0 |
| nsw19 | 4 | 3 | .1910 | .0394 | 11.0 | 33.3 |
| nsw20 | 2 | 16 | .3016 | .0275 | 21.625 | 36.0 |
| nsw20 | 3 | 2 | .4325 | .1135 | 4.0 | 7.0 |
| nsw22 | 2 | 2 | .0058 | .0263 | 38.0 | 12.0 |
| nsw22 | 3 | 1 | .0062 | .0459 | 10.0 | 6.0 |
| nsw22 | 4 | 3 | .0139 | .0704 | 47.333 | 12.333 |

At 10 km, GLM gives higher average positive probability in 3/4 eligible nsw19 folds and 2/2 nsw20 folds; Bayesian gives higher average positive probability in all 3 eligible nsw22 folds. nsw20 fold 2 also has higher mean negative prediction under GLM (.0398) than Bayesian (.0085), so positive sensitivity must not be interpreted without false-positive/calibration tradeoffs.

**Cautions:** Mean positive probability alone is not calibration, and positive mean rank does not establish a universal AP advantage. Very few positives per fold and strong concentration in nsw20 limit stability. These are within-dataset, spatially buffered OOF predictions and descriptive diagnostics, not external validation or proof of a causal modeling mechanism. See [fold counts](RARE_TAXON_FOLD_INSTABILITY_VERIFIED.md), [LOO sensitivity](RARE_TAXON_LOO_SENSITIVITY_VERIFIED.md), and [eight-taxon bootstrap](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md).

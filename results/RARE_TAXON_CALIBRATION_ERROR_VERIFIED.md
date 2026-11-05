# Verified rare-taxon probability error and threshold audit

[Successful CI #42101184615](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42101184615), artifact `rare-taxon-calibration-error-audit` (fold and pooled aggregate CSVs); code [scripts/80_rare_taxon_calibration_error_audit.py](../scripts/80_rare_taxon_calibration_error_audit.py). Matched OOF predictions for quadratic GLM and Bayesian models at 5/10/20 km. All public outputs aggregate across records.

## 10 km quadratic models, 1,167 held-out records per taxon

| Taxon | Model | Positives | Mean predicted p | Observed prevalence | Brier | Log loss | Predicted p >= 0.10: TP | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nsw19 | Bayesian | 8 | .007397 | .006855 | .006706 | .032564 | 0 | 0 |
| nsw19 | GLM | 8 | .008188 | .006855 | .007751 | .029962 | 2 | 21 |
| nsw20 | Bayesian | 18 | .006119 | .015424 | .014711 | .062023 | 1 | 8 |
| nsw20 | GLM | 18 | .013445 | .015424 | .015880 | .080374 | 8 | 15 |
| nsw22 | Bayesian | 6 | .005090 | .005141 | .004763 | .022666 | 1 | 1 |
| nsw22 | GLM | 6 | .004183 | .005141 | .005995 | .038521 | 0 | 10 |

At the exploratory probability threshold of 0.10, nsw20 GLM identifies 8/18 detections with 15 false positives, while Bayesian identifies 1/18 with 8 false positives. Bayesian nevertheless has lower Brier and log loss. nsw19 shows metric disagreement: Bayesian has lower Brier but GLM lower log loss. nsw22 favors Bayesian on both scores, but only 6 positive records exist.

**Interpretation limits:** This is a threshold-specific diagnostic, not an optimized operating point. Overall mean predicted probability versus prevalence is calibration-in-the-large, **not** a reliability curve or complete calibration validation. Rare-taxon results have very small positive counts, are from the same underlying dataset, and do not establish universal model superiority. Do not claim significance from these descriptive contrasts; see [eight-taxon bootstrap](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md), [fold instability](RARE_TAXON_FOLD_INSTABILITY_VERIFIED.md), and [positive rank diagnostics](RARE_TAXON_POSITIVE_RECORD_DIAGNOSTICS_VERIFIED.md).

# Manuscript-ready results: spatially held-out ecological SDM comparison

## Study sample and validation

The analysis includes 9,336 species–plot binary records, representing eight tree species across 1,167 plots. The three-model matched evaluation compares Bayesian hierarchical partial pooling, species-specific MaxEnt (`maxnet`), and XGBoost on identical held-out observations for the five species with evaluable spatial folds. Three very rare species lack sufficient evaluable folds and are excluded from this matched ranking. The matched dataset contains 23 species–fold evaluations. The comparison is descriptive; these results do not constitute a paired significance test.

## Matched species-level performance

Values are reproduced from `results/matched_bayesian_maxent_xgboost_species.csv`. Species-level results are unweighted means over eligible folds, rather than pooled prediction-level scores.

| Species | Eligible folds | Positives | Bayesian AUC | MaxEnt AUC | XGBoost AUC | Bayesian AP | MaxEnt AP | XGBoost AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Angophora costata* | 5 | 244 | 0.7912 | 0.8219 | **0.8645** | 0.4036 | 0.4618 | **0.5536** |
| *Eucalyptus blakelyi* | 3 | 39 | 0.7289 | **0.7693** | 0.7550 | 0.0997 | 0.1352 | **0.1391** |
| *Eucalyptus campanulata* | 5 | 109 | 0.7119 | 0.7597 | **0.8085** | 0.1658 | 0.2126 | **0.3078** |
| *Eucalyptus fastigata* | 5 | 270 | 0.8084 | **0.8136** | 0.7909 | **0.5384** | 0.5374 | 0.4728 |
| *Eucalyptus nova-anglica* | 5 | 53 | 0.7520 | 0.7613 | **0.7705** | 0.1381 | **0.1494** | 0.1469 |

XGBoost has the largest mean AUC for three of five evaluable species; MaxEnt for two; Bayesian for none. Average precision yields a different ranking for some species, underscoring the importance of evaluating minority-class detection alongside ROC AUC. The differences should not be described as statistically significant without appropriate uncertainty estimates.

## Paired spatial-fold differences

Across 23 identical species–fold evaluations, the mean within-fold AUC difference was +0.0406 for XGBoost minus Bayesian, +0.0255 for MaxEnt minus Bayesian, and +0.0151 for XGBoost minus MaxEnt. Corresponding mean AP differences were +0.0563, +0.0297, and +0.0266. XGBoost exceeded Bayesian on AUC in 17/23 matched combinations, MaxEnt exceeded Bayesian in 18/23, and XGBoost exceeded MaxEnt in 14/23. These comparisons are **descriptive**: the folds are dependent, and the results must not be interpreted as confidence intervals or significance tests. Full species-specific differences are provided in [the paired-differences audit](MATCHED_MODEL_PAIRED_DIFFERENCES.md).

## Bayesian convergence confirmation

The original Bayesian fit used two chains and 1,200 iterations per chain (600 warmup). Independent confirmation fits used four chains and 2,000 iterations per chain (1,000 warmup). Across all five confirmation folds, all 175 parameter–fold diagnostic rows passed R-hat <= 1.01 and bulk/tail ESS >= 400. Twenty chains had zero divergences and zero treedepth-limit hits; minimum BFMI was 0.645. See `results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md`.

The confirmed-fit fold AUCs are 0.8818, 0.9104, 0.8682, 0.9028, and 0.9037, respectively. Absolute changes from original-fit fold AUCs are all below 0.00033. These are **fold-level all-species metrics**, not directly comparable with the matched five-species AUC values above. Original-fit held-out predictions underpin the three-model matched comparison; confirmation artifacts do not contain per-record predictions.

## Interpretation and limitations

The spatially held-out results indicate substantial species-to-species heterogeneity. The tree-based model performs best on AUC for three species, whereas the presence-only/background-based MaxEnt benchmark performs best for two. Bayesian partial pooling provides a coherent framework for shared effects and species-level uncertainty but does not automatically maximize discrimination on sparse held-out positives. The three rarest species cannot be ranked reliably using the available folds. Comparisons should be treated as exploratory because fold eligibility varies by species, the data are geographically restricted, and no independent external-region validation or inferential uncertainty on model differences is provided.

**Figure references:** `results/figures/22_three_model_auc.svg` and `results/figures/22_three_model_ap.svg`.

**Reproducibility:** `results/matched_bayesian_maxent_xgboost_species.csv`, `results/matched_bayesian_maxent_xgboost_folds.csv`, `results/matched_three_model_plot_data.csv`, `results/bayesian_five_fold_confirmed_metrics.csv`, and the two historical Actions runs 41279140198 and 41277076588.

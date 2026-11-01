# Manuscript-ready results: spatially held-out ecological SDM comparison

## Study sample and validation

The analysis includes 9,336 species–plot binary records, representing eight tree species across 1,167 plots. The three-model matched evaluation compares Bayesian hierarchical partial pooling, species-specific MaxEnt (`maxnet`), and XGBoost on identical held-out observations for the five species with evaluable spatial folds. Three very rare species lack sufficient evaluable folds and are excluded from this matched ranking. The matched dataset contains 23 species–fold evaluations. The comparison is descriptive; these results do not constitute a paired significance test.

The [species-by-fold presence audit](SPECIES_SPATIAL_FOLD_PRESENCE_AUDIT.md) confirms that *Corymbia gummifera* has 8 presences distributed across three folds (4/1/3), *Corymbia intermedia* has 18 across two folds (9/9), and *Eucalyptus carnea* has six across two folds (1/5). Two folds for *Eucalyptus blakelyi* have no positives, leaving three evaluable folds. Zero-positive held-out folds cannot yield defined ROC AUC; rare-species exclusion from the matched benchmark is a separate eligibility choice, not evidence that all their folds are unscorable.

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

## Weighting sensitivity

Because one species contributed only three evaluable folds while the others contributed five, we also averaged paired differences with equal weight per species rather than per species–fold. The equal-species AUC differences were +0.0394 (XGBoost minus Bayesian), +0.0267 (MaxEnt minus Bayesian), and +0.0127 (XGBoost minus MaxEnt); the corresponding AP differences were +0.0549, +0.0302, and +0.0248. All contrasts retained their sign under this reweighting. This deterministic sensitivity analysis does not establish inferential significance; see [weighting sensitivity details](MATCHED_MODEL_WEIGHTING_SENSITIVITY.md).

## Spatial-fold ranking heterogeneity

Across 23 eligible matched species–fold comparisons, XGBoost ranked first in ROC AUC on 14 folds, MaxEnt on seven and Bayesian on two; the corresponding AP first-place counts were 13, seven and three. Rankings varied across spatial folds within species: for *Eucalyptus fastigata*, AUC wins were split Bayesian two, MaxEnt one and XGBoost two. Thus, species-mean model rankings should not be interpreted as consistent dominance in every held-out region. See [fold-level ranking audit](MATCHED_MODEL_FOLD_HETEROGENEITY.md). These counts are descriptive and do not imply independent trials.

## Bayesian convergence confirmation

The original Bayesian fit used two chains and 1,200 iterations per chain (600 warmup). Independent confirmation fits used four chains and 2,000 iterations per chain (1,000 warmup). Across all five confirmation folds, all 175 parameter–fold diagnostic rows passed R-hat <= 1.01 and bulk/tail ESS >= 400. Twenty chains had zero divergences and zero treedepth-limit hits; minimum BFMI was 0.645. See `results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md`.

The confirmed-fit fold AUCs are 0.8818, 0.9104, 0.8682, 0.9028, and 0.9037, respectively. Absolute changes from original-fit fold AUCs are all below 0.00033. These are **fold-level all-species metrics**, not directly comparable with the matched five-species AUC values above. Original-fit held-out predictions underpin the three-model matched comparison; confirmation artifacts do not contain per-record predictions.

## Interpretation and limitations

The spatially held-out results indicate substantial species-to-species heterogeneity. The tree-based model performs best on AUC for three species, whereas the presence-only/background-based MaxEnt benchmark performs best for two. Bayesian partial pooling provides a coherent framework for shared effects and species-level uncertainty but does not automatically maximize discrimination on sparse held-out positives. The three rarest species cannot be ranked reliably using the available folds. Comparisons should be treated as exploratory because fold eligibility varies by species, the data are geographically restricted, and no independent external-region validation or inferential uncertainty on model differences is provided.

**Publication figures:** [ROC AUC](figures/23_three_model_auc_publication.svg) and [average precision](figures/23_three_model_ap_publication.svg). The earlier [AUC](figures/22_three_model_auc.svg) and [AP](figures/22_three_model_ap.svg) charts are retained for comparison.

**Figure 1 caption (ROC AUC).** Mean held-out ROC AUC for Bayesian hierarchical, MaxEnt and XGBoost models on identical spatial-fold observations within five eligible species. Bars represent unweighted species-specific means of eligible fold scores (23 species–fold combinations total); annotations report eligible folds and positive observations. Values are descriptive, without confidence intervals or significance tests. Bayesian values use original-fit predictions.

**Figure 2 caption (average precision).** Mean held-out average precision for the same matched species–fold combinations. Species differ in positive-class prevalence, and AP should be interpreted with this imbalance in mind. Bars are unweighted species-specific fold means, not confidence intervals. Bayesian values use original-fit predictions.

**Reproducibility:** `results/matched_bayesian_maxent_xgboost_species.csv`, `results/matched_bayesian_maxent_xgboost_folds.csv`, `results/matched_three_model_plot_data.csv`, `results/bayesian_five_fold_confirmed_metrics.csv`, and the two historical Actions runs 41279140198 and 41277076588.

## Supplementary analysis: rare-species survey prioritization

Using existing original-fit Bayesian out-of-fold probabilities, we evaluated within-fold rankings of three sparse species under fixed survey budgets of 10, 25 and 50 plots per spatial fold. The five folds collectively contain 1,167 plots; selecting 25 per fold therefore inspects 125 plots (10.7% of all plots). A random-selection reference was calculated analytically as the sum across folds of k times observed fold prevalence, rather than through a stochastic simulation.

| Species | Observed presences | Top-25-per-fold captured | Random expected captures | Descriptive enrichment |
|---|---:|---:|---:|---:|
| *Corymbia gummifera* | 8 | 3 | 0.849 | 3.53× |
| *Corymbia intermedia* | 18 | 17 | 1.911 | 8.90× |
| *Eucalyptus carnea* | 6 | 4 | 0.643 | 6.22× |

For *C. intermedia*, the model retrieved eight of nine presences in fold 1 and all nine in fold 3 among the top 25 plots in each fold. These descriptive results suggest possible value for prioritizing limited surveys, but performance is heterogeneous and based on very few positive cases. Zero-presence folds remain included in the survey budget; they are not silently discarded. The random baseline represents an expected count, not a p-value, uncertainty interval or alternative fitted model. Importantly, pooled across-fold rankings and fold-local rankings address different deployment strategies and must not be conflated.

**Supplementary Figure S1 caption.** Observed presence recall among the highest-scored 10, 25 or 50 plots *within each of five geographic holdout folds*, contrasted with the expected recall from randomly selecting the same number of plots in each fold. Each panel is one rare species, and denominators are eight, 18 and six observed presences respectively. Predictions come from original shorter-chain Bayesian spatial cross-validation; results are descriptive and not independent external validation. [Supplementary Figure S1 (SVG)](figures/24_rare_species_retrieval_publication.svg) · [Per-fold retrieval](bayesian_rare_species_within_fold_retrieval.csv) · [Random baseline](bayesian_rare_species_random_baseline.csv).

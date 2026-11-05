## Buffered spatial validation: GLM-only sensitivity

The [verified alternative-partition buffer refits](results/BUFFERED_GLM_ALTERNATIVE_SPATIAL_VERIFIED.md) excluded training plots within 0, 5, 10, or 20 projected-distance km of any held-out plot and refitted both per-taxon regularized GLMs. Quadratic AUC changed **0.9022 → 0.9027 → 0.8994 → 0.8913**; linear AUC **0.8892 → 0.8894 → 0.8864 → 0.8771**. Quadratic GLM retained higher pooled AUC/AP at every radius, while absolute discrimination fell at 20 km. Training-set shrinkage and unverified source CRS metadata limit causal interpretation; these are **GLM-only within-dataset results**, not a matched buffered Bayesian comparison or external validation.

## Taxon-cluster uncertainty for the four-model comparison

A [verified 20,000-resample eight-taxon bootstrap](results/FOUR_MODEL_TAXON_BOOTSTRAP_VERIFIED.md) found GLM quadratic minus Bayesian quadratic equal-taxon mean ΔAUC **+0.0064** (exploratory 95% interval **−0.0267 to +0.0397**) and ΔAP **+0.0277** (**−0.0418 to +0.1218**). All reported intervals for GLM–Bayesian and quadratic–linear contrasts cross zero. **Pooled model ranking does not establish a stable advantage across taxa**; intervals are exploratory with only eight taxa and are not confirmatory significance tests.

## Verified four-model comparison on identical spatial holdouts

The [matched four-model audit](results/FOUR_MODEL_ALTERNATIVE_SPATIAL_VERIFIED.md) verified identical 9,336 held-out plot–taxon records for per-taxon L2-regularized GLM linear/quadratic and hierarchical Bayesian linear/quadratic on the alternative spatial five-fold partition. **Quadratic GLM** achieved the highest pooled AUC (**0.9022**) and AP (**0.4284**); **quadratic Bayesian** achieved the lowest pooled Brier (**0.05591**) and log loss (**0.18730**). Across 34 eligible taxon–fold cells, GLM quadratic minus Bayesian quadratic mean ΔAUC was **+0.00683** and ΔAP **+0.02460**; across 40 cells ΔBrier was **+0.00031**. These are exploratory within-dataset descriptive differences, not independent external validation or proof of universal superiority. Taxon heterogeneity remains a central limitation.

## Taxon influence: important qualification

A [verified second-partition taxon-exclusion audit](results/SECOND_PARTITION_TAXON_EXCLUSION_VERIFIED.md) shows the pooled quadratic advantage is **not uniform across taxa**. With all eight taxa, pooled quadratic-minus-linear differences were ΔAUC **+0.0143**, ΔAP **+0.0727**, and ΔBrier **−0.00647**. When **both nsw18 and nsw24** are excluded from evaluation, the differences reverse (ΔAUC **−0.0069**, ΔAP **−0.0314**, ΔBrier **+0.00028**). These are evaluation-subset exclusions, **not retraining without the taxa**. The defensible conclusion is taxon-dependent nonlinear benefits, not universal quadratic superiority.

## Alternative-partition diagnostics (verified)

The [second-partition extended audit](results/SECOND_PARTITION_EXTENDED_DIAGNOSTICS_VERIFIED.md) passed all MCMC thresholds across ten fits (max Rhat **1.0082**, min bulk/tail ESS **814/1,412**, min BFMI **0.705**, zero divergences or treedepth hits). Quadratic versus linear improved pooled Brier by **0.00647** and ECE10 by **0.00581**, but Brier improved for only **5/8 anonymous taxa**, with much of the gain from nsw18 and nsw24. Mean held-out residual five-nearest-neighbor Moran's I fell **0.2815 → 0.2471** (24/40 cells improved); residual spatial autocorrelation persists. All are within-dataset sensitivity results, not external validation.

## Alternative spatial partition robustness (November 2026)

A second outcome-blind spatial partition (15 geographic clusters into five folds; adjusted Rand index 0.0955 versus the original partition) was used to **refit both linear and quadratic hierarchical Bayesian models**, 10 successful four-chain jobs in total. The [verified paired audit](results/ALTERNATIVE_SPATIAL_BAYESIAN_VERIFIED.md) found 34 eligible taxon–fold AUC/AP cells: mean paired ΔAUC **+0.01675**, ΔAP **+0.02299**; across all 40 cells, ΔBrier **−0.00648** and ΔLog loss **−0.01741**. Pooled AUC was **0.8856 → 0.8999** and pooled AP **0.3360 → 0.4087**. These are favorable **within-dataset alternative-partition** results, **not external validation**.

## Verified Bayesian specification extension (November 2026)

A five-fold Bayesian sensitivity analysis compared the archived hierarchical baseline with quadratic rainfall/temperature responses and a nonpooled taxon-specific model. This is **within-dataset model-specification validation, not an independent external cohort**. The original three-pipeline comparison uses 23 eligible matched taxon–fold cells; the expanded Bayesian AUC/AP analysis has 30 eligible cells.

On the **original 23 cells**, quadratic Bayesian improved mean paired AUC by **+0.0453**, AP by **+0.0831**, and Brier by **−0.0118** relative to archived Bayesian predictions. In the expanded analysis, mean paired AUC was **+0.0378** and AP **+0.0813**. Across all held-out records, pooled Brier improved **0.06084 → 0.05404**, 10-bin ECE **0.01985 → 0.01255**, and mean residual 5-NN Moran's I **0.341 → 0.291**. Positive residual spatial autocorrelation remains. Excluding nsw18, the quadratic gains retained favorable directions. These results are exploratory, and the baseline OOF predictions came from a shorter-chain fit.

See [consolidated verified findings](results/BAYESIAN_EXTENSION_CONSOLIDATED_VERIFIED.md) for evidence, limitations and links to successful CI runs. The manuscript remains private.

# Ecological Species Distribution Modeling — NSW, Australia

**Research portfolio | R · XGBoost · MaxEnt · Bayesian hierarchical modeling · Spatial validation**

[**View the visual case study**](https://ys30.github.io/projects/ecological-sdm-xgboost.html) · [**Inspect reproducible results**](results/matched_bayesian_maxent_xgboost_folds.csv) · [**Historical analyses**](results/README_HISTORICAL_DETAILED.md)

## The research question

How robust are species distribution model rankings when evaluated on the **same geographically held-out records**, and can their predictions support **fixed-budget surveys of sparsely observed taxa**?

**Data:** Public `disdat` NSW open-forest benchmark: **1,167 plots**, **8 anonymized taxa**, **9,336 plot–taxon observations**, **747 recorded detections**. Observed non-detections are not detection-corrected absences.

## Geographic study design — Figure 1

![Figure 1. NSW geographic survey locations and five held-out spatial folds](results/figures/figure1_geographic_nsw_holdouts.png)

**Figure 1.** Geographic context and plot-level five-fold holdout assignments. A 4×4 longitude/latitude quantile grid allocates entire cells to folds. **No distance buffer:** 218/1,167 held-out plots were within 5 km of training plots; 37 were within 1 km. Results characterize withheld plots **within NSW**, not an independent-region transfer experiment.

## Matched three-model results

Identical held-out evaluation records: **23 eligible taxon–fold cells across five taxa**.

| Implemented pipeline | Mean ROC AUC | Distinctive modeling assumption |
|---|---:|---|
| **Pooled XGBoost** | **0.8016** | Gradient-boosted trees; taxon identity and environmental/categorical predictors |
| **Taxon-specific MaxEnt** | 0.7865 | Regularized suitability from four environmental predictors; surveyed non-detections as background approximation |
| **Hierarchical Bayesian SDM** | 0.7610 | Bernoulli occurrence model; partial pooling of taxon intercepts and selected slopes |

**Mean paired XGBoost–MaxEnt AUC difference: +0.0151.** This is a *descriptive pipeline comparison*, **not** an isolated algorithm effect: models have unequal information sets and score interpretations. The archived XGBoost benchmark uses a **fixed 300-round configuration**, not nested spatially tuned hyperparameters.

![Matched taxon-level ROC AUC for XGBoost, MaxEnt and Bayesian pipelines](results/figures/23_three_model_auc_publication.svg)

[Matched species metrics](results/matched_bayesian_maxent_xgboost_species.csv) · [Matched fold metrics](results/matched_bayesian_maxent_xgboost_folds.csv) · [Average precision figure](results/figures/23_three_model_ap_publication.svg)

### Spatial-fold heterogeneity — paired differences

![XGBoost minus MaxEnt ROC AUC, with individual spatial folds and taxon means](results/figures/25_paired_xgboost_maxent_auc.svg)

![XGBoost minus MaxEnt average precision, with individual spatial folds and taxon means](results/figures/26_paired_xgboost_maxent_ap.svg)

XGBoost exceeded MaxEnt in **14/23 eligible ROC AUC cells** (mean paired ΔAUC **+0.01508**) and **13/23 AP cells** (mean paired ΔAP **+0.02658**). The direction varies by taxon and fold. Points are descriptive; the 23 cells are not independent replicates. [Figure methods and audit caveats](results/MATCHED_PAIRED_FIGURES_AND_METHOD_AUDIT.md).

## What changes the model ranking?

**Fold-local XGBoost feature encoding** has now passed CI and its CSV outputs are archived. Across the 23 matched cells, XGBoost with the common four environmental predictors alone had mean ΔAUC **−0.1652** versus MaxEnt; adding taxon identity changed that to **+0.0163**, while the fold-local full encoding gave **+0.0125**. These are descriptive pipeline contrasts: removing taxon identity changes the prediction task and does not isolate an algorithm effect. [Final verified spatial robustness and encoding audit](results/EMS_SPATIAL_ROBUSTNESS_FINAL.md).

[Feature-representation sensitivity script](scripts/41_xgboost_spatial_encoding_sensitivity.R) · [Bayesian five-fold convergence audit](results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md)

## Practical use: rare-taxon survey prioritization

**Why evaluate rare taxa differently?** Three taxa had too few positives in some held-out folds to support the same per-fold ROC AUC comparison as the five evaluable taxa. Rather than exclude them, we asked a practical question: *with a fixed number of plots to resurvey, how many previously recorded detections would a model-ranked shortlist capture?*

**Evaluation protocol:** In each of the five held-out geographic folds, rank plots using the hierarchical Bayesian model and select the top 10, 25 or 50. Count observed detections in those selections and compare with **fold-specific random selection** at the identical budget. This is retrospective screening, not prospective validation.

**At top 25 plots per fold:** 125 / 1,167 plots screened (10.7%).

| Anonymous taxon | Recorded detections | Captured | Recall | Expected under random | Enrichment |
|---|---:|---:|---:|---:|---:|
| `nsw19` | 8 | 3 | 37.5% | 0.85 | **3.5×** |
| `nsw20` | 18 | 17 | 94.4% | 1.91 | **8.9×** |
| `nsw22` | 6 | 4 | 66.7% | 0.64 | **6.2×** |

![Rare-taxon retrieval across top-10, top-25 and top-50 budgets, with anonymous taxon labels](results/figures/24_rare_species_retrieval_publication.svg)

**Interpretation:** The model concentrated known detections more effectively than random selection in this historical survey, especially for `nsw20`. Performance was heterogeneous: `nsw19` still missed five of eight recorded detections at top 25. With only 6–18 positives per taxon, percentages are sensitive to individual observations; the enrichment factors are descriptive, not estimates with demonstrated prospective generalizability.

**Decision value:** Ranked shortlists could guide *where to investigate first* when field resources are constrained. Deployment requires explicit survey cost and access constraints, spatial coverage, detection-probability modeling and prospective independent evaluation. Observed non-detection is not proof of absence, and retrospective recall is not expected future discovery yield.

[Fold-level top-k data](results/bayesian_rare_species_within_fold_retrieval.csv) · [Fold-specific random baseline and enrichment](results/bayesian_rare_species_random_baseline.csv)

## Matched Bayesian vs XGBoost rare-taxon retrieval

A second, pipeline-level comparison uses **identical held-out spatial folds, anonymous taxa and inspection budgets**. Here, XGBoost is the **taxon-specific four-environment-predictor** pipeline; the earlier figure above is Bayesian-only.

![Bayesian and taxon-specific XGBoost recall across matched fold-local top-k budgets](results/figures/27_rare_taxon_matched_topk_bayesian_xgboost.svg)

| Taxon | Observed presences | Top 25/fold: Bayesian | Top 25/fold: XGBoost |
|---|---:|---:|---:|
| `nsw19` | 8 | 3/8 (37.5%) | **7/8 (87.5%)** |
| `nsw20` | 18 | **17/18 (94.4%)** | 16/18 (88.9%) |
| `nsw22` | 6 | **4/6 (66.7%)** | 1/6 (16.7%) |

**What matters:** the preferred model changes by taxon and inspection budget. A single average AUC would obscure the very different screening outcomes for `nsw19` and `nsw22`. These are retrospective, within-region **pipeline comparisons**, not independent field trials or algorithm-only causal effects.

[Verified matched counts and interpretation](results/RARE_TAXON_MATCHED_BAYES_XGBOOST_VERIFIED.md) · [Machine-readable fold-matched comparison](results/rare_taxon_bayesian_xgboost_matched_topk.csv) · [Reproduction script](scripts/54_compare_rare_taxon_retrieval.py)

## Common-input XGBoost ablation: taxon identity matters

Across eight taxa, the equal-taxon mean **pooled minus taxon-specific** AUC difference was **−0.0123** (exploratory taxon-bootstrap range −0.0510 to +0.0220). Removing species identity from the pooled model decreased AUC by **0.3131** on the same common four-environment-predictor experiment. These findings distinguish model architecture from the information supplied to it; they do not establish independent cross-region generalization.

[Verified common-input results](results/EMS_V614_COMMON_INPUT_VERIFIED_RESULTS.md)

**Metric eligibility audit:** the archived v6.14 table has 40 taxon–fold rows, of which 30 contain both outcome classes and support AUC/AP; the other 10 are single-class folds with undefined AUC/AP but valid Brier scores. The source table labels all 40 rows `ok`; this is a status-label limitation, not evidence that AUC/AP were calculated in single-class folds. [Verified row-level eligibility audit](results/v614_common_input/metric_eligibility_audit.csv).

**Spatial-distance robustness:** in 21 eligible paired taxon–fold cells across five taxa, the XGBoost-minus-MaxEnt AUC advantage changed by **−0.0915** (cell mean) between near (<5 km) and far (≥5 km) held-out plots. The equal-taxon mean was **−0.0835**, with exploratory taxon sign-flip **p = 0.125**; this is not evidence of a confirmed interaction or independent geographic transfer. [Detailed paired audit](results/EMS_SPATIAL_ROBUSTNESS_FINAL.md).

## Reproducibility and limitations

- **Spatial evaluation:** five unbuffered geographic folds, with plots kept intact across taxa.
- **Predictor comparability:** XGBoost uses more information than MaxEnt and Bayesian models; differences should not be presented as algorithm-only effects.
- **Bayesian diagnostics:** four-chain confirmation fits passed archived convergence thresholds; matched figures use original-fit Bayesian predictions because complete long-chain observation-level prediction artifacts were not retained.
- **Practical interpretation:** results support within-region ranking and retrospective screening, not calibrated occupancy, independent geographic transfer or a prospective field survey.

**The current journal manuscript and appendix are managed separately from the public results.** Historical manuscript-like draft files and older analyses remain accessible in this public repository; they have not been migrated or deleted. Different validation protocols and historical materials are linked from the [historical README](results/README_HISTORICAL_DETAILED.md), not mixed into the current matched-model headline.

**Map provenance:** [GitHub Actions geographic Figure 1 build](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41540075797).


**New verified buffered Bayesian–GLM comparison (5/10/20 km):** all 30 Bayesian fits passed and the matched 9,336-record OOF audit succeeded ([run #41917257355](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41917257355)). Across all three radii, quadratic GLM has higher pooled AUC/AP while quadratic hierarchical Bayesian has lower Brier/log loss; see [verified results and limitations](results/BUFFERED_FOUR_MODEL_MATCHED_VERIFIED.md). This is within-dataset spatial sensitivity, not external validation; coordinate units remain to be CRS-verified.

**Buffered model uncertainty (verified):** 20,000 equal-taxon bootstrap resamples across eight anonymized taxa ([run #41930995688](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41930995688)). All 24 GLM-minus-Bayesian 95% descriptive percentile intervals (linear/quadratic × 5/10/20 km × four metrics) include zero. [Full verified summary](results/BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md). No statistically significant superiority claim.

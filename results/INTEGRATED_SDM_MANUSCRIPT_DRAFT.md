# Spatially held-out evaluation of XGBoost, MaxEnt and hierarchical Bayesian models for multi-species tree occurrence prediction

**Status:** Integrated manuscript draft; not yet journal-submission ready. Data source is author-confirmed as Valavi et al. (2023); author list, affiliations, survey dates, source license and final hyperparameter audit remain to be verified.

**Foundational methods references:** The Bayesian hierarchical implementation is contextualized by Bürkner (2017); average precision is reported alongside ROC AUC given rare positives (Saito & Rehmsmeier, 2015); `maxnet` and cloglog background are documented by Phillips et al. (2017). The study data source is Valavi et al. (2023), as confirmed by the dataset provider/author; the row-level reconstruction audit is a separate technical verification. Two successful GitHub Actions audits show that all 1,167 local plot environmental signatures match the `disdat` NSW open-forest reference data; exact coordinate-and-species-label identity was subsequently confirmed in GitHub Actions run 41287426532.

**Additional foundational methods references:** The spatial-validation rationale follows Roberts et al. (2017) and Valavi et al. (2019); the algorithmic context follows Phillips et al. (2006) for MaxEnt and Chen and Guestrin (2016) for XGBoost. The project implements its own geographic quantile-grid fold assignment rather than invoking the `blockCV` R package. [Verified citations and DOIs](VERIFIED_METHODS_REFERENCES.md).

## Data-source provenance: verified and pending evidence

Two independently completed GitHub Actions runs ([41286199826](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41286199826) and [41286053395](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41286053395)) installed the `disdat` package and compared the local data against `disdat::disPa("NSW","ot")` and `disdat::disEnv("NSW","ot")`. Across the six common environmental variables (`disturb`, `rainann`, `soildepth`, `soilfert`, `tempann`, `topo`), **all 1,167 local plot-level signatures were found among the reference environmental records** (1,167/1,167). The reference presence–absence and environmental tables each contained 2,075 sites. This initial environmental comparison was subsequently strengthened by exact coordinate, site-ID and all 9,336 binary-label checks in the completed identity audit. The reference uses species codes `nsw18`–`nsw25`, whereas the working table uses taxonomic names.

A completed independent identity audit ([GitHub Actions run 41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532), artifact `nsw-disdat-site-label-identity`) matched **1,167/1,167 plot coordinates** in both reference PA and ENV tables, confirmed exact equality for **all six environmental variables**, verified consistent reference site IDs, and found a **unique eight-species code mapping with zero mismatches across all 9,336 binary observations** (`FULL_IDENTITY_CONFIRMED: TRUE`). Thus the analytical table is confirmed to correspond to the `disdat` NSW open-forest survey records sourced from Valavi et al. (2023). This analysis performs **internal spatial cross-validation on those survey observations**, not independent external validation on a separately collected survey. The source license and survey collection dates remain to be checked against original documentation.

## Abstract

**Background:** Species distribution models can achieve strong discrimination under conventional cross-validation while transferring less reliably to geographically distinct observations. Sparse species create additional challenges for comparative evaluation and survey prioritization.

**Methods:** We evaluated XGBoost, species-specific MaxEnt and a hierarchical Bayesian occurrence model using five two-dimensional spatial holdout folds. The dataset comprised 9,336 binary species–plot records for eight tree species at 1,167 plots. The matched three-model benchmark included five species with 23 eligible species–fold comparisons. For three rarer species, a separate Bayesian-only analysis assessed out-of-fold probability errors and presence retrieval at fixed per-fold survey budgets.

**Results:** XGBoost achieved the highest species-mean ROC AUC for three of five evaluable species, and MaxEnt for two. Across the 23 matched species–fold comparisons, the descriptive mean AUC differences were +0.0406 (XGBoost minus Bayesian), +0.0255 (MaxEnt minus Bayesian) and +0.0151 (XGBoost minus MaxEnt). Model rankings varied by species and spatial fold. Independent longer-chain Bayesian fits passed convergence and sampler diagnostic thresholds across all five folds. In a distinct rare-species analysis, Bayesian within-fold prioritization of 25 plots per fold (125/1,167 plots total) retrieved 17 of 18 observed *Corymbia intermedia* presences, compared with 1.91 expected under geographically stratified random selection; findings for other rare species were more variable.

**Conclusions:** Geographic holdouts reveal meaningful species- and region-dependent variation in algorithm performance. Sparse-species survey prioritization is a promising exploratory application, but the present evidence does not establish model superiority, prospective survey gains or external geographic generalizability.

**Keywords:** species distribution models; spatial cross-validation; ecological monitoring; XGBoost; MaxEnt; Bayesian hierarchical model; rare species; survey prioritization.

## 1. Introduction

Predictive species distribution models (SDMs) are frequently used to describe environmental associations, map potential occurrence and guide biodiversity monitoring. Their utility for decisions in new locations depends not only on predictive discrimination but also on how well evaluation procedures approximate geographic transfer. Spatially structured training and test observations can make random partitions overly optimistic when nearby plots share environmental and ecological characteristics.

Tree occurrence modeling provides a useful setting in which to compare flexible machine-learning algorithms, ecological niche models and hierarchical statistical models. Gradient-boosted trees can capture nonlinear predictor relationships and interactions; MaxEnt offers a widely used presence/background-oriented modeling framework; and Bayesian hierarchical models allow information sharing among species while representing parameter and predictive uncertainty. These approaches optimize different objectives and therefore need not agree on species rankings or performance under spatial holdout.

A further challenge is the uneven occurrence prevalence of species. When positives are rare, a held-out geographic block can contain no observed presences, rendering ROC AUC undefined. Even when a block contains one positive, rankings may be highly unstable. Rather than interpret missing ROC AUC as model failure, an alternative is to distinguish comparative algorithm benchmarking from probability scoring and survey-prioritization analyses for sparse species.

We therefore asked: (1) How do XGBoost, MaxEnt and a hierarchical Bayesian model compare on identical spatially held-out observations for evaluable tree species? (2) How stable are these comparisons across species and geographic folds? (3) Can Bayesian out-of-fold probabilities provide descriptive prioritization of survey locations for species too sparse for a reliable three-model AUC ranking? We deliberately report descriptive comparisons rather than treating correlated species–fold observations as independent inferential replications.

## 2. Materials and methods

### 2.1 Occurrence observations and geographic holdouts

The analysis comprised eight tree species measured as binary occurrence at 1,167 plots, producing 9,336 species–plot records (747 observed presences). The committed table records longitudes of 151.2229–153.3644°E and latitudes of 32.7068–28.3287°S, consistent with eastern New South Wales, Australia. Each plot has one binary record for each of the eight species. The original survey provider, years, ecological non-detection protocol, projected coordinate reference system and environmental predictor measurement units have not yet been independently established; consequently, no unsupported physical units are assigned to the rainfall, temperature, soil-depth or topography fields. See the [source-data geometry and units audit](SOURCE_DATA_GEOMETRY_AND_UNITS_AUDIT.md). Spatial blocks were constructed from median longitude and latitude per plot, with quantile breakpoints forming an approximately 4 × 4 geographic grid. Occupied grid cells (rather than individual plots) were assigned to five folds by a reproducible shuffled greedy algorithm balancing plot counts (random seed 2026). All records from a plot remained in the same fold; response labels were not used to form blocks or allocate folds. This is a spatial-block stress test, **not** buffered spatial cross-validation or independent external-region validation: geographically adjacent training and validation cells can remain close. The fold construction is implemented in `scripts/11_spatial_2d_block_cv.R`. The same fold assignments were used for the matched model comparison. Eligible species–fold cells required both observed presences and absences in the held-out fold; five species yielded 23 eligible cells. Three sparse species were analyzed separately rather than assigned an unstable overall three-model AUC rank. The spatial block construction and species-specific presence counts are audited in `results/SPECIES_SPATIAL_FOLD_PRESENCE_AUDIT.md`.

### 2.2 Predictive models

The three approaches were XGBoost, species-specific MaxEnt using `maxnet`, and a hierarchical Bayesian Bernoulli-logit model using `brms`. The Bayesian model included annual rainfall, annual temperature, soil depth and topographic predictors, with fold-specific scaling estimated using training data only. Species-level varying intercepts and varying rainfall and temperature slopes allowed partial pooling; correlations among varying effects were not estimated. Reported priors were Normal(-2.5, 2) for the population intercept, Normal(0, 1) for population slopes and Exponential(2) for varying-effect standard deviations. The XGBoost spatial-block implementation used a binary logistic objective with log-loss evaluation, learning rate 0.03, maximum tree depth 4, row subsampling 0.8, column subsampling 0.8, two threads and 300 boosting rounds per fold (fixed rather than fold-tuned). Its design matrix included annual rainfall, soil depth, annual temperature, topography, species identity, disturbance and soil fertility; the latter three were represented as factors in an R model matrix. Coordinates were used for spatial partitioning but excluded from model predictors. The species-specific MaxEnt implementation used annual rainfall, soil depth, annual temperature and topography only, linear and quadratic feature classes (`classes='lq'`), and regularization multiplier 2. Training presences were contrasted with all surveyed non-detection records for the focal species in training folds; these non-detections are a background approximation, not independently sampled landscape background. MaxEnt predictions used the cloglog output as a relative suitability ranking and were not treated as calibrated occurrence probabilities. MaxEnt folds were skipped when training presences or training background numbered fewer than 10, or when the held-out fold contained only one response class. Model-specific source files are `scripts/11_spatial_2d_block_cv.R` and `scripts/13_maxent_spatial_2d_cv.R`; matched XGBoost–MaxEnt records were joined and checked in `scripts/14_matched_maxent_xgboost.R`. Exact software versions and environmental dataset provenance remain to be documented.

### 2.3 Matched predictive evaluation

Held-out predictions were compared on identical observations for eligible species–fold combinations. We report within-fold receiver-operating-characteristic area under the curve (ROC AUC) and average precision (AP), followed by arithmetic means across eligible folds within species. The paired contrast summary gives unweighted averages across 23 eligible species–fold cells. An equal-species-weighted sensitivity analysis addresses the fact that one species contributed only three eligible folds while four contributed five. Fold winner counts summarize geographic heterogeneity. These are descriptive quantities; correlated folds and species invalidate a naive independent-cell significance interpretation.

### 2.4 Bayesian diagnostic confirmation

The original Bayesian fits used two chains, 1,200 iterations per chain and 600 warmup iterations. Independent confirmation fits used four chains, 2,000 iterations and 1,000 warmup iterations. We checked R-hat <= 1.01, bulk and tail effective sample size >= 400, divergences, treedepth-limit hits and BFMI. The original-fit held-out predictions were used for matched algorithm comparisons and rare-species retrieval. Historical long-chain confirmation artifacts do not provide per-record predictions, so their fold-level AUC stability cannot establish identical individual rankings.

### 2.5 Sparse-species probability scoring and retrieval

For *Corymbia gummifera*, *C. intermedia* and *Eucalyptus carnea*, original-fit Bayesian out-of-fold probabilities were evaluated using Brier score and Bernoulli log loss. A baseline predicted each held-out fold using the species prevalence estimated from the other four training folds only. Presence retrieval was evaluated in two different ways: a pooled ranking of all 1,167 plots per species and a geographically stratified ranking allocating k plots separately to each of five spatial folds. For the latter, k=10, 25 or 50 screens 5k plots overall. The expected number of presences under random within-fold selection was calculated as Σ_f kP_f/N_f, with P_f observed positives and N_f plots in fold f. This random reference is an analytical expectation, not a fitted model, significance test or independent prospective survey.

## 3. Results

### 3.1 Species prevalence and benchmark eligibility

Three sparse species were omitted from the five-species matched ranking because positives were geographically uneven: *C. gummifera* had eight presences across three folds (4, 1 and 3), *C. intermedia* had 18 across two folds (9 and 9), and *E. carnea* had six across two folds (1 and 5). Zero-positive folds cannot yield ROC AUC, and one-positive folds provide fragile estimates. *Eucalyptus blakelyi* contributed three eligible folds rather than five.

### 3.2 Matched three-model performance

| Species | Folds | Bayesian AUC | MaxEnt AUC | XGBoost AUC | Bayesian AP | MaxEnt AP | XGBoost AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Angophora costata* | 5 | 0.7912 | 0.8219 | 0.8645 | 0.4036 | 0.4618 | 0.5536 |
| *Eucalyptus blakelyi* | 3 | 0.7289 | 0.7693 | 0.7550 | 0.0997 | 0.1352 | 0.1391 |
| *Eucalyptus campanulata* | 5 | 0.7119 | 0.7597 | 0.8085 | 0.1658 | 0.2126 | 0.3078 |
| *Eucalyptus fastigata* | 5 | 0.8084 | 0.8136 | 0.7909 | 0.5384 | 0.5374 | 0.4728 |
| *Eucalyptus nova-anglica* | 5 | 0.7520 | 0.7613 | 0.7705 | 0.1381 | 0.1494 | 0.1469 |

XGBoost led species-mean AUC for three species and MaxEnt for two. AP rankings were not identical to AUC rankings, including for *E. blakelyi* and *E. nova-anglica*. Across 23 matched cells, mean AUC differences were +0.0406 for XGBoost versus Bayesian, +0.0255 for MaxEnt versus Bayesian and +0.0151 for XGBoost versus MaxEnt. Corresponding mean AP differences were +0.0563, +0.0297 and +0.0266. Equal-species weighting preserved the direction of the AUC differences (+0.0394, +0.0267 and +0.0127). No inferential significance is claimed.

### 3.3 Geographic heterogeneity and Bayesian confirmation

Across eligible species–fold cells, AUC first-place counts were 14 XGBoost, seven MaxEnt and two Bayesian; AP first-place counts were 13, seven and three. The relative ranking varied within species across folds. The five independent long-chain Bayesian fits passed all 175 parameter–fold R-hat and effective-sample-size diagnostic checks. Twenty chains had no divergences or treedepth-limit hits, and minimum BFMI was approximately 0.645. Long-chain fold-level all-species AUCs were 0.8818, 0.9104, 0.8682, 0.9028 and 0.9037, with absolute deviations from the original fits below 0.00033.

### 3.4 Rare-species probability errors

| Species | Bayesian Brier | Training-prevalence Brier | Bayesian log loss | Training-prevalence log loss |
|---|---:|---:|---:|---:|
| *C. gummifera* | 0.007294 | 0.006835 | 0.039273 | 0.043137 |
| *C. intermedia* | 0.012959 | 0.015385 | 0.045149 | 0.086924 |
| *E. carnea* | 0.005033 | 0.005154 | 0.026652 | 0.038934 |

Bayesian log loss was lower than the training-only prevalence reference for all three species, whereas Brier score was worse for *C. gummifera*. Probability error alone did not consistently imply strong retrieval of rare positives.

### 3.5 Survey prioritization under geographic allocation

| Species | Presences | Top 25 per fold captured | Random expected captures | Descriptive enrichment |
|---|---:|---:|---:|---:|
| *C. gummifera* | 8 | 3 | 0.849 | 3.53× |
| *C. intermedia* | 18 | 17 | 1.911 | 8.90× |
| *E. carnea* | 6 | 4 | 0.643 | 6.22× |

The top-25-per-fold strategy screens 125 plots in total, 10.7% of 1,167. For *C. intermedia*, eight of nine presences in fold 1 and all nine in fold 3 were retrieved within their fold-local top 25. A pooled top-25 ranking screens only 25 plots overall and must not be conflated with the five-fold allocation. The results are exploratory, especially given six to 18 positives per species.

## 4. Discussion

### 4.1 Geographic transfer changes model interpretation

The matched spatial evaluation revealed species-specific algorithm rankings rather than universal superiority of a single approach. XGBoost was strongest on species-mean AUC for three species, whereas MaxEnt was strongest for two. This pattern is consistent with model-specific sensitivity to ecological response structure and geographic sampling, although the present study cannot identify these as causal mechanisms. AUC and AP answer different predictive questions, and both are useful when presences are scarce.

### 4.2 Hierarchical modeling and computational reliability

The Bayesian model supports partial pooling and predictive uncertainty, but its species-mean discrimination did not lead the matched benchmark. Improved convergence diagnostics from longer chains support computational reliability of the fitted Bayesian models, not ecological calibration or superiority. Because long-chain per-record predictions were not archived, the original predictions remain the basis for all matched model and survey-prioritization findings.

### 4.3 Rare-species survey design as a distinct application

The fold-stratified retrieval exercise suggests that Bayesian probabilities may help target surveys under a specified geographic allocation, particularly for *C. intermedia*. However, the per-fold and pooled rankings spend different survey budgets and reflect different allocation policies. Several geographic folds have zero observed presences; sampling there still consumes resources. The reported enrichment factors compare observed captures with a random expectation and cannot be interpreted as confidence intervals, statistical significance or future field performance.

### 4.4 Limitations

The analysis is geographically restricted, has only five species eligible for comparative AUC ranking, and has very few presences for the three rare species. Species–fold evaluations are correlated and unequal in eligibility; no independent external-region test or spatially appropriate uncertainty interval for paired differences is available. Occupancy, imperfect detection, survey access costs and sampling bias were not explicitly modeled. Model-specific hyperparameters, environmental-data provenance and geographic study-area details still require an author-verified methods audit. The Bayesian-only rare-species results cannot establish the performance of MaxEnt or XGBoost for those species.

## 5. Conclusions

A common spatial-holdout evaluation framework revealed heterogeneity in the discrimination of XGBoost, MaxEnt and hierarchical Bayesian models across tree species and geographic regions. While XGBoost achieved the highest mean AUC for three of five eligible species, no method consistently dominated across all species and folds. A separate Bayesian-only rare-species extension demonstrated descriptive enrichment of observed presences under a fixed per-region survey budget, without establishing prospective operational gains. These findings support reporting both geographically stratified predictive performance and explicitly budgeted monitoring metrics, while maintaining a clear distinction between exploratory model comparisons and validated ecological decisions.

## Figures and supplementary materials

- **Figure 1:** [Matched species-level ROC AUC](figures/23_three_model_auc_publication.svg). Five species, eligible spatial folds, original Bayesian predictions; no uncertainty intervals.
- **Figure 2:** [Matched species-level average precision](figures/23_three_model_ap_publication.svg). Same observations and folds as Figure 1.
- **Supplementary Figure S1:** [Rare-species within-fold retrieval and random expected recall](figures/24_rare_species_retrieval_publication.svg). Three rare species, 10/25/50 plots selected separately within each of five folds; original Bayesian predictions.
- **Supplementary Table S1:** [Matched species–fold scores](matched_bayesian_maxent_xgboost_folds.csv).
- **Supplementary Table S2:** [Rare-species fold-level retrieval](bayesian_rare_species_within_fold_retrieval.csv).
- **Supplementary Table S3:** [Random selection expected captures](bayesian_rare_species_random_baseline.csv).
- **Reproducibility:** [Rare-species source-to-figure audit](RARE_SPECIES_RETRIEVAL_REPRODUCIBILITY_AUDIT.md); [Bayesian convergence confirmation](BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md).

## Evidence and editorial verification

The manuscript claims have been mapped to their source files in the [pre-submission verification matrix](SDM_PRE_SUBMISSION_VERIFICATION_MATRIX.md). In particular, the model-specific configuration and predictor provenance are not yet fully documented; these should be verified against executable code and dataset metadata before inserting detailed methodological claims. The rare-species retrieval calculations and figure were independently cross-checked against committed predictions and CSVs, but the current end-to-end GitHub Actions run has not been confirmed as successful.

## Author completion checklist (required before submission)

- [ ] Confirm study area, coordinate reference system, geographic sampling frame and environmental predictor sources.
- [ ] Confirm original species survey dataset provenance, permissions and any public-release restrictions.
- [x] Verify XGBoost fixed spatial-CV parameters, MaxEnt feature classes/background strategy and matching code against executable scripts.
- [ ] Record exact R, xgboost, maxnet, brms and dependency versions for the computational environment.
- [ ] Add verified literature references, in-text citations and target-journal formatting.
- [ ] Confirm author names, order, affiliations, funding, conflicts, data availability and code availability statements.
- [ ] Verify final GitHub Actions runs; independent data consistency checks alone are not CI confirmation.
- [ ] Decide whether spatially appropriate uncertainty analyses or external validation are needed for the target journal.


## References (verified foundational sources)

Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models using Stan. *Journal of Statistical Software*, 80(1), 1–28. https://doi.org/10.18637/jss.v080.i01

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., & Blair, M. E. (2017). Opening the black box: An open-source release of Maxent. *Ecography*, 40(7), 887–893. https://doi.org/10.1111/ecog.03049

Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2023). Flexible species distribution modelling methods perform well on spatially separated testing data. *Global Ecology and Biogeography*, 32, 369–383. https://doi.org/10.1111/geb.13639



Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10(2), 225–232. https://doi.org/10.1111/2041-210X.13107

Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*, 190(3–4), 231–259. https://doi.org/10.1016/j.ecolmodel.2005.03.026

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

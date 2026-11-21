# v6.23 manuscript integration draft — matched spatial validation

**Status:** Working draft for author review. This is not the frozen v6.14 manuscript, not a submitted manuscript, and not evidence of external validation.

## Working title

When model complexity helps: matched spatially buffered evaluation of gradient boosting and logistic regression across eight ecological taxa

## Abstract (structured working draft)

**Background:** Ecological species-detection models can appear to benefit from nonlinear machine learning, but comparisons are difficult to interpret when models use different spatial folds, training exclusions, or evaluation metrics. Rare taxa further complicate model selection.

**Methods:** We compared XGBoost, linear logistic regression (GLM), and quadratic GLM on eight taxa (1,167 records per taxon; 9,336 held-out predictions per model). All models used the same five outer spatial folds and a 10-km training exclusion buffer. Each taxon was scored on its pooled outer out-of-fold (OOF) predictions, then given equal weight in the cross-taxon summary. We assessed log loss, Brier score, AUC and average precision (AP), plus calibration-in-the-large, observed-prevalence stratification, leave-one-taxon-out and leave-one-outer-fold-out sensitivity analyses.

**Results:** Equal-taxon mean log loss was 0.186905 for XGBoost, 0.204591 for linear GLM and 0.187524 for quadratic GLM; corresponding Brier scores were 0.055984, 0.062737 and 0.054642. Mean absolute calibration-in-the-large bias was 0.002923, 0.004033 and 0.003887, respectively. XGBoost's log-loss and Brier advantage over linear GLM retained its direction after removing any one taxon or any one held-out spatial fold. The XGBoost–quadratic GLM log-loss difference was small (−0.000619) and reversed direction in one taxon-deletion and one fold-deletion analysis. The four lower-prevalence taxa had 6–39 detections each, compared with 53–270 among the remaining four taxa.

**Conclusions:** Under a matched spatially buffered protocol, gradient boosting improved proper probability scores relative to a linear GLM but did not consistently dominate a quadratic GLM. Conclusions depended on comparator complexity, scoring rule and prevalence. These descriptive findings require independent validation and should not be interpreted as significance tests.

## Reproducible model configuration (verified code; insert into Methods §2.3)

The XGBoost classifier used a binary logistic objective, histogram tree construction, 100 trees, maximum depth 2, learning rate 0.03, minimum child weight 1, row subsampling 0.8, column subsampling 0.8 and random seed 622. Its predictors were `rainann`, `soildepth`, `tempann` and `topo`, on their original numeric scales. Both logistic regressions used L2-regularized `LogisticRegression` with `C=100`, `lbfgs`, maximum 3,000 iterations and convergence tolerance `1e-8`. Their `StandardScaler` was fitted on eligible training rows only. The linear GLM used all four standardized predictors; the quadratic GLM additionally used squared standardized `rainann` and `tempann`, without pairwise interactions. The experiment used fixed configurations, not inner-loop or nested hyperparameter optimization. See `docs/V623_EXACT_MODEL_SPECIFICATION.md` for the detailed source-linked description.

**Updated verification (2026-11-20):** the full three-model comparison, manuscript-number gate, figure technical gate, and actual supplement artifact inventory all passed in [CI #44780027918](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44780027918). The inventory includes file-level SHA-256 digests and CSV row counts. Technical image checks are not a substitute for human visual review. Data licensing, original survey metadata, authoritative predictor units and upstream leakage remain unresolved.

## 1. Introduction

Ecological presence–absence models are increasingly used to summarize species–environment relationships and to estimate detection or occurrence probabilities across heterogeneous landscapes. Flexible machine-learning algorithms can capture nonlinear associations, but their apparent predictive advantage depends on both the comparator and the validation design. In particular, a comparison against a strictly linear baseline may attribute to algorithm choice an improvement that a modest, prespecified nonlinear statistical model could also achieve. A fair assessment therefore needs a common prediction target, the same held-out observations, and training sets defined by the same spatial exclusion rules. [CITATION NEEDED: ecological SDM model comparison and nonlinear baselines]

Spatial dependence creates an additional challenge. Nearby training and test plots can share environmental conditions and unmeasured processes, making randomly partitioned validation optimistic for some intended uses. Spatial blocking and a distance-based exclusion buffer reduce direct geographic proximity between training and test observations, although neither guarantees independence nor establishes transfer performance in a different region or year. The interpretation of a validation score must remain tied to the geographic sampling design actually evaluated. [CITATION NEEDED: spatial cross-validation and transferability]

Model evaluation also depends on the quantity that matters for the ecological decision. Log loss and the Brier score are proper probability scoring rules, whereas area under the receiver operating characteristic curve (AUC) and average precision (AP) emphasize ranking performance. These measures can disagree, especially for uncommon outcomes. A small mean calibration-in-the-large bias does not imply that predictions are reliable throughout the probability range, and a taxon with very few observed events can have unstable subgroup or fold-level diagnostics. Consequently, reporting several complementary metrics is more informative than declaring a universal winner based on one summary. [CITATION NEEDED: proper scoring rules, discrimination, calibration and rare outcomes]

We evaluated a fixed-configuration gradient-boosted tree model (XGBoost) against two logistic regression comparators, one linear and one containing prespecified quadratic terms, for eight anonymized ecological taxa. All three models shared five spatial outer folds, a 10-km exclusion buffer around each held-out fold, and the same test records. Each taxon contributed equally to the primary cross-taxon summaries. We asked whether XGBoost's advantage over a linear model persisted against a modest nonlinear baseline, whether conclusions changed with the scoring rule or observed event prevalence, and whether any one taxon or held-out fold disproportionately influenced the descriptive contrasts. Our objective was a reproducible within-dataset comparison, not a claim of externally validated transferability or statistically established algorithm superiority.

## 2. Methods — verified design and points to audit before submission

### 2.1 Dataset and outcomes
Eight taxa, each with 1,167 records. The number of detected events varies markedly (6–270 per taxon). **Source identity is now verified:** archived NSW `disdat` site-and-label identity audit matched 1,167/1,167 geographic sites, all six environmental covariates and all 9,336 binary labels (CI #41287426532; `results/NSW_DISDAT_EXECUTED_PROVENANCE_AUDIT.md`). The reference is attributed there to Valavi et al. (2023). The analyzed longitude/latitude range is 151.2229–153.3644°E and 32.7068–28.3287°S (`results/SOURCE_DATA_GEOMETRY_AND_UNITS_AUDIT.md`). **Still required before submission:** original field survey dates and sampling design, authoritative covariate units (especially rainfall), data reuse license, privacy permissions, missing-data rules and any class balancing. Anonymous species codes are retained; do not disclose underlying taxonomic names without permission.

### 2.2 Matched spatially buffered outer validation
Five primary outer spatial folds with a 10-km exclusion buffer for training observations. The matched comparison checks row identities, labels, folds and data/fold hashes. In the verified full rerun, both XGBoost and GLM scripts independently emitted SHA-256 hashes of their actual train/test row-ID sets for each taxon and outer fold; a separate audit reconstructed the eligible sets from the source data and fold map and confirmed equality across all 80 model-variant comparison cells. This establishes identical row membership, but not identical feature transformations (GLM uses fold-local scaling while XGBoost uses raw predictors). Each model produces 9,336 pooled OOF predictions (one per record). **Author must confirm in Methods:** spatial grouping algorithm, distance computation, predictor preprocessing inside each training fold, inner tuning scope, exact hyperparameter selection and seeds. Do not label the comparison fully nested or fully tuned unless the run provenance proves it.

### 2.3 Models
All three classifiers used four predictors (`rainann`, `soildepth`, `tempann` and `topo`), with original measurement units still requiring source confirmation. The XGBoost binary logistic classifier used histogram-based tree construction, 100 trees, maximum depth 2, learning rate 0.03, minimum child weight 1, row and per-tree column subsampling both 0.8, and random seed 622. Predictors entered XGBoost without standardization.

Both logistic regressions used scikit-learn's L2-regularized `LogisticRegression` (`C=100`, `lbfgs`, `max_iter=3000`, `tol=1e-8`, `random_state=20261104`). A `StandardScaler` was fitted only on eligible training observations in each outer fold and applied to the corresponding held-out observations. The linear model included all four standardized predictors. The quadratic model additionally included squared standardized annual rainfall and annual temperature, without pairwise interactions or squared soil-depth/topography terms. All three models clipped probabilities to [1e-12, 1−1e-12] for numerical stability. Configurations were fixed, with no nested or inner-loop hyperparameter search in this matched comparison. A historical Bayesian model is **not** a fourth matched comparator because its training protocol did not share these exclusions.


### 2.4 Evaluation
For each taxon, compute log loss, Brier, AUC and AP from pooled OOF predictions, then average across taxa with equal weight. Lower log loss and Brier are preferable; higher AUC and AP are preferable. Ten of 40 individual taxon-by-fold test cells had one class, so fold-level AUC/AP may be undefined; pooled taxon metrics are the primary discrimination summaries. Calibration-in-the-large is mean predicted probability minus observed prevalence, evaluated per taxon before macro-averaging. Reliability deciles are equal-count groups **within each model and taxon**.

### 2.5 Descriptive sensitivity analyses
Split taxa into four lower- and four higher-prevalence groups based on observed detection counts; this is an exploratory, outcome-defined split, not a preregistered ecological threshold. Recompute equal-taxon contrasts after omitting each taxon in turn. Separately, omit one outer held-out fold's OOF test records at a time **without retraining**; recompute per-taxon pooled log loss and Brier on remaining rows. Neither procedure produces a confidence interval or an independent spatial replicate.

## 3. Results

### 3.1 Overall matched model performance

| Model | Equal-taxon log loss ↓ | Equal-taxon Brier ↓ | Mean absolute calibration-in-the-large bias ↓ |
|---|---:|---:|---:|
| XGBoost | **0.186905** | 0.055984 | **0.002923** |
| Linear GLM | 0.204591 | 0.062737 | 0.004033 |
| Quadratic GLM | 0.187524 | **0.054642** | 0.003887 |

XGBoost minus linear GLM was −0.017686 for log loss and −0.006753 for Brier. Against quadratic GLM, the corresponding differences were −0.000619 and +0.001341. The latter two contrasts favor different models depending on the proper scoring rule.

For discrimination, XGBoost minus linear GLM was −0.013164 for AUC and +0.023991 for AP; XGBoost minus quadratic GLM was −0.021906 for AUC and −0.016299 for AP. These are equal-taxon pooled OOF contrasts, not fold-average scores or inferential tests.

### 3.2 Observed-prevalence stratification
The four lower-prevalence taxa had 6–39 positives each, while the other four had 53–270. Relative to linear GLM, XGBoost's log-loss improvement was −0.002174 in the lower-prevalence group versus −0.033197 in the higher-prevalence group. Against quadratic GLM, XGBoost's log-loss contrast was −0.011809 versus +0.010571, respectively. With four taxa per group and a data-defined split, these are exploratory patterns rather than confirmed effect modification.

### 3.3 Taxon influence
After omitting each of eight taxa in turn, XGBoost minus linear GLM log loss ranged from −0.022554 to −0.005312, and Brier from −0.008236 to −0.002003, without sign reversals. Against quadratic GLM, log loss ranged from −0.006134 to +0.005858, with one sign reversal.

### 3.4 Spatial outer-fold influence
After omitting each of five spatial test folds in turn, XGBoost minus linear GLM log loss ranged from −0.021376 to −0.014430 and Brier from −0.008726 to −0.005572, with no sign reversals. Against quadratic GLM, log loss ranged from −0.003030 to +0.005290 and reversed once; Brier remained positive (+0.000557 to +0.001951). These ranges do not represent uncertainty intervals.

### 3.5 Calibration
The mean signed calibration-in-the-large biases were +0.001365 (XGBoost), +0.003877 (linear GLM) and +0.003384 (quadratic GLM), suggesting slight average overprediction. XGBoost had the smallest mean absolute calibration-in-the-large bias, but quadratic GLM had the best Brier score. The full-range and low-probability reliability figures are available as CI artifacts; visual inspection is still needed before asserting whether local calibration is satisfactory.

## 4. Discussion

### 4.1 The comparator changes the interpretation of model complexity

Under identical outer folds and 10-km training exclusions, XGBoost achieved lower equal-taxon mean log loss (0.186905) than the linear logistic regression (0.204591), with a corresponding Brier improvement (0.055984 versus 0.062737). The log-loss contrast remained in the same direction in all eight leave-one-taxon-out and five leave-one-held-out-fold-out summaries. These results establish a consistent **descriptive** advantage over this particular linear baseline within the sampled plots. They do not establish that gradient boosting is intrinsically superior to logistic regression or that the difference would persist in independent surveys.

The interpretation changes when the comparator includes two prespecified quadratic terms. The quadratic GLM achieved a mean log loss of 0.187524, only 0.000619 higher than XGBoost, while attaining a lower mean Brier score (0.054642 versus 0.055984). Moreover, the XGBoost-minus-quadratic-GLM log-loss contrast changed sign after removing one taxon and, separately, after removing one held-out spatial fold. Thus, the strongest conclusion is not a uniform advantage for the more flexible algorithm, but that the measured advantage depends materially on the statistical baseline and on the scoring rule. A relatively small addition of nonlinear terms was sufficient to narrow the log-loss gap in this dataset.

### 4.2 Proper scores, ranking and calibration answer different questions

AUC and AP do not measure the same aspect of predictions as log loss and Brier score. Here, the equal-taxon AUC contrast favored both GLM comparators over XGBoost, whereas the AP contrast favored XGBoost over the linear GLM but not over the quadratic GLM. This divergence is not an inconsistency: probability accuracy and rank ordering are distinct prediction goals. Which model is preferable therefore depends on whether an ecological application requires well-calibrated probabilities, ranking of candidate sites, or a thresholded decision supported by explicit costs and benefits.

XGBoost had the smallest mean absolute calibration-in-the-large bias (0.002923), but that single summary cannot establish calibration at low predicted probabilities or across individual taxa. The quadratic GLM's better Brier score also cautions against interpreting average calibration bias as a complete calibration assessment. Full-range and low-probability reliability displays should be considered alongside the numerical summaries, with particular attention to the number of observed events contributing to each bin.

### 4.3 Rare taxa and heterogeneous outcomes

Four taxa had only 6–39 observed detections among 1,167 records each, compared with 53–270 detections for the other four. The descriptive log-loss advantage of XGBoost over the linear GLM was much smaller among the lower-prevalence taxa (−0.002174) than among the higher-prevalence taxa (−0.033197). Against the quadratic GLM, the contrast differed in direction between the two prevalence groups (−0.011809 and +0.010571). These patterns are useful for identifying where aggregate model rankings may conceal heterogeneity, but they do not establish a prevalence-by-model interaction. The split was defined using the observed outcomes, with only four taxa per group, and no independent validation was performed.

Sparse detections also complicate diagnostic interpretation. Some individual held-out taxon–fold cells contained only one outcome class, making fold-specific AUC or AP undefined. The primary discrimination metrics were therefore calculated from each taxon's pooled out-of-fold predictions, not from averages over potentially undefined fold-level scores. Reliability bins containing few or no detections should be interpreted cautiously rather than treated as precise evidence of local calibration.

### 4.4 Spatial validation, reproducibility and limits of inference

The shared spatial folds and 10-km buffer address an important source of unfairness in algorithm comparisons: different models did not benefit from different test records or geographically closer training observations. The original fold allocation was independently replayed, and exact fit-time train/test observation identifiers were audited against reconstructed memberships. These checks strengthen the reproducibility of the **comparison protocol**, but do not show that the sampled plots are independent or that upstream site selection and covariate construction were free of target leakage.

The leave-one-taxon-out and leave-one-held-out-fold-out analyses were influence diagnostics, not bootstrap replicates, spatial confidence intervals, or formal tests of superiority. Omission of a fold re-scores existing held-out predictions without retraining the models. The findings are therefore restricted to the sampled NSW data and the fixed configurations evaluated here. A future assessment of geographic or temporal transfer would require genuinely independent regions, surveys or years and a prespecified external validation design.

### 4.5 Implications and outstanding limitations

For ecological model selection, these results support reporting a plausible nonlinear statistical comparator, matched geographic exclusions, multiple probability and ranking metrics, and taxon-level rather than solely pooled summaries. They also argue against interpreting small differences in one score as a universal hierarchy of model families. Before submission, the original survey dates and design, authoritative predictor units (particularly `rainann`), source licensing and any upstream data-selection leakage must be resolved and documented. The eight taxa remain anonymized, and historical Bayesian analyses are not treated as a matched fourth arm. The current study provides a reproducible within-dataset comparison rather than external validation or confirmatory inference.

## 5. Figure and supplement plan

- **Main Figure 1:** Matched per-taxon scores, already generated by `scripts/110_v623_manuscript_figures.py`; visually inspect axis scaling and readability.
- **Main Figure 2:** Paired proper-score differences, already generated; distinguish log loss and Brier units.
- **Supplementary Figure S1:** Eight-taxon reliability panels, full 0–1 scale, generated by `scripts/115_v623_reliability_figures.py`.
- **Supplementary Figure S2:** Low-probability 0–0.3 reliability zoom; disclose clipping explicitly.
- **Supplementary Tables:** 64 leave-one-taxon-out contrasts; 20 leave-one-spatial-fold-out proper-score contrasts; 240 reliability decile rows; 24 taxon-model calibration summaries; one-class fold diagnostics.

## Reproducibility and provenance

Primary workflow: `.github/workflows/v623-matched-three-model.yml`.
- Matched model comparison: `scripts/109_v623_matched_three_model_comparison.py`.
- Rare-taxon stratification: `scripts/111_v623_rare_taxon_sensitivity.py`.
- Taxon influence: `scripts/112_v623_leave_one_taxon_out.py`.
- Calibration: `scripts/113_v623_matched_calibration.py`.
- Fold influence: `scripts/114_v623_fold_level_sensitivity.py`.
- Reliability figures: `scripts/115_v623_reliability_figures.py`.

CI-verified runs: [three-model comparison](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44428289030), [calibration macro summaries](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44451628271), [spatial fold influence](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44451175135), [reliability figures](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44452488604).

**Publication gate:** complete original survey metadata, license and environmental units; inspect figures; audit upstream leakage; decide a defensible uncertainty approach before calling this submission-ready. Training-set equality and geographic fold replay have passed independent CI checks.

## Verified fit-time membership audit (2026-11-18)

Full rerun [#44463218295](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44463218295) passed Python compile checks, 40 XGBoost fits, 80 GLM fits, both training-exclusion and exact fit-time ID-hash reconstruction audits, and artifact upload. This supports the precise statement that **the same train/test row-ID sets were used for the three models within each outer fold and taxon**. It does not prove the absence of upstream leakage in the construction of predictors or spatial folds. The comparison remains **fixed configuration**, not nested hyperparameter tuning.

## v6.23 provenance and publication consistency update (2026-11-19)

The dataset's NSW `disdat` identity is **verified**, not merely suspected: archived [CI #41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532) established site, covariate and response-label agreement. Later [CI #44525359745](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44525359745) passed matched-model refitting, exact fit-time ID-hash checks and a fold-local numeric-feature/scaler audit. The original fold-generation code (`scripts/11_spatial_2d_block_cv.R`) constructs a 4×4 longitude/latitude quantile grid and allocates whole geographic blocks to five folds without using response labels in the assignment algorithm. **These checks do not establish that upstream site selection, covariate creation or preprocessing were leakage-free.**

### Publication checklist, in priority order

1. Resolve original data redistribution rights, source survey dates and the exact primary-source citation.
2. Resolve the environmental-variable data dictionary and units, particularly `rainann`. The source-geometry report cites provisional definitions of `tempann` (°C ×10), `soildepth` (m ×1000) and `topo` (m); do not silently reinterpret raw values.
3. **Spatial-fold replay completed:** pinned R 4.4.3 workflow [#44681382736](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44681382736) successfully reconstructed the archived mapping. **Still open:** audit upstream target-dependent feature selection and joins.
4. **Analysis protocol frozen:** see `docs/V623_VALIDATED_PROTOCOL_FREEZE.md` and successful full comparison [#44678216771](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44678216771). Still freeze manuscript-ready tables and figures from one named run, visually inspect all figures, and harmonize captions with Results.
5. Retain the fixed-configuration, within-dataset and non-significance-test limitations. Do not merge historical Bayesian results into the matched three-model numerical claims.

## Draft manuscript figure captions (protocol-frozen; visual QC pending)

**Figure 1.** Out-of-fold predictive performance for eight anonymous taxa under identical five-fold spatial validation with a 10-km training exclusion buffer. Panels show taxon-level log loss, Brier score, ROC AUC and average precision for fixed-configuration XGBoost, linear logistic GLM and quadratic logistic GLM. Each point represents pooled outer-fold predictions for one taxon. The models share outer test observations and eligible training row identities. Values are descriptive; the figure does not display confidence intervals. ROC AUC and average precision should be interpreted in light of the markedly different event prevalences.

**Figure 2.** Within-taxon paired differences in log loss and Brier score (XGBoost minus GLM) for the same held-out observations and outer folds. Negative values favor XGBoost; positive values favor the indicated GLM. Zero is the no-difference reference line. Contrasts with linear and quadratic GLMs are shown separately; do not equate differences in log loss with differences in Brier units. No significance tests or spatially valid uncertainty intervals are implied.

**Supplementary Figure S1.** Taxon-specific out-of-fold reliability diagrams on the full predicted-probability range, comparing XGBoost with both GLMs. Observed event frequency is shown against binned predicted probability; sparse bins for low-prevalence taxa may be highly unstable. Calibration-in-the-large alone does not establish local reliability.

**Supplementary Figure S2.** The same taxon-specific reliability diagnostics restricted to predicted probabilities between 0 and 0.3. The restricted x-axis is a visualization choice and must not be interpreted as full-range calibration; the full-range counterpart is provided in Supplementary Figure S1.

**Editorial gate:** Caption text above is ready for layout review, **not proof that the generated plots were visually inspected**. Check legends, axis limits, sample sizes, panel labels, resolution and accessibility against the successful CI artifact before final submission.

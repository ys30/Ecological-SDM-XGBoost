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

## 1. Introduction — argument outline

Ecological prediction is vulnerable to spatially optimistic validation and to comparisons that conflate algorithm choice with evaluation design. Proper scoring rules, ranking metrics and calibration address different properties of predictions; no single metric establishes a universal winner. Very low detection counts can also destabilize discrimination and reliability estimates. Our goal is a transparent, matched-protocol comparison, not a claim that a particular algorithm is always best.

**Research questions:** (1) Does XGBoost outperform a linear GLM when outer folds and 10-km training exclusions are identical? (2) Does the conclusion survive comparison with a quadratic GLM? (3) How do rarity and the choice of evaluation metric affect the interpretation? (4) Are observed contrasts sensitive to a single taxon or held-out spatial fold?

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

## 4. Discussion — working interpretation

**Comparator choice matters.** The apparent advantage of XGBoost over a linear GLM diminishes when the baseline can express quadratic effects. This supports the general methodological point that machine-learning superiority claims should include plausible nonlinear statistical baselines.

**Prediction goals matter.** Log loss, Brier, AUC and AP quantify different properties. Here the ranking and probability metrics do not identify one universally dominant model. Model choice should be aligned with the intended ecological decision.

**Rarity matters for interpretation.** With as few as six detections per taxon, reliability deciles frequently have no positive cases and per-fold discrimination can be undefined. The prevalence split is exploratory, and a small average calibration bias cannot establish good local calibration.

**Spatial dependence limits inference.** The matched 10-km exclusion protocol is stronger than a naive random split, but the reported five-fold deletion ranges are not spatially valid confidence intervals. No independent geographic or temporal transfer test is demonstrated here.

**Scope and limitations.** Report the exact source, region, observation design, class balance, predictor definitions, training exclusions and tuning provenance before submission. The Bayesian historical analysis is complementary rather than a matched fourth model. Avoid inferential language about prevalence modification or superiority without an appropriate spatial uncertainty design.

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

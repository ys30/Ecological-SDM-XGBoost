# EMS v6.22 — prospective supplementary analysis plan
Status: PRE-FREEZE. This document applies only to the prospective v6.22 supplementary study, not to the historical v6.13–v6.21 exploratory analyses.

## Research question
Compare effects of response specification, information sharing, model family, and tuning under spatially held-out evaluation. The 19 models and 20 contrasts are defined in config/v622_model_matrix.csv and config/v622_contrasts.csv; those CSVs are authoritative for identifiers and pairings. Do not describe these observational comparisons as isolated causal effects: models can differ in loss functions, flexibility, pooling assumptions and response semantics.

## Data and provenance
The base source commit is 4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d. Six pinned input hashes appear in config/v622_experiment.yaml. The study uses 1,167 plots, eight anonymised taxa (nsw18–nsw25), 9,336 plot–taxon records and 747 detections. Preserve spatial assignments and exclusions from pinned inputs. Do not silently substitute regenerated splits or altered data.

## Primary evaluation
Run grid4_s2026_b0 and km15_b0 as co-primary spatial schemes. For each outer fold, inner tuning uses 12 spatial kmeans clusters constructed using *only outer training plots*, grouped into four inner folds. Outer held-out data are never used for tuning or calibration. Nested spatial tuning is required for the tuned XGBoost, glmnet and MaxEnt configurations. XGBoost uses 30 fixed-seed sampled hyperparameter configurations with no early stopping. MaxEnt tuning searches regularisation and feature classes; glmnet searches alpha and lambda. All transformations, scaling, null prevalence estimates and fallback choices must be derived only from the relevant training split.

## Model matrix and contrasts
The 19-row matrix includes per-taxon and pooled XGBoost, GLM and shared GLM, GLMM, archived Bayesian fits, MaxEnt and glmnet. The added glmnet_quad_rt uses quadratic rainfall and temperature with the same four raw environmental predictors. Contrast C2f compares glmnet_quad_env against glmnet_quad_rt, separating additional quadratic terms from the rain/temperature quadratic specification. The 20-row contrasts file fixes primary versus secondary comparison tiers. Do not silently alter contrast definitions after freeze.

## Eligibility and failures
Taxon-level training groups with fewer than five positives use a training-prevalence constant predictor, flagged as fallback. An optimisation failure receives one documented retry; if unsuccessful, use the documented fallback and log it. AUC cells require both classes and no fallback in either paired model. Report coverage and exclusions for every comparison; do not infer absence of effect from unevaluable cells.

## Metrics and tuning
Primary outer evaluation uses within-taxon-fold paired ROC AUC differences, with secondary average precision, log loss, Brier score and coverage. For probabilistic inner tuning, maximise equal-taxon average relative log-loss skill, 1 - LL_model/LL_null. Estimate null prevalence with (positives + 0.5)/(n + 1) on the inner-training data; clip probabilities to [1e-6, 1-1e-6]. For MaxEnt, inner within-taxon AUC is the selection metric; its cloglog log loss is sensitivity only because presence-background outputs are not automatically interchangeable with presence-absence probabilities.

## Inference and decisions
**Primary uncertainty:** 2,000 taxon-cluster bootstrap resamples, seed 622, percentile 95% intervals. Mixed-effects estimates are auxiliary; they are not the primary confidence intervals. Primary contrasts are evaluated separately in both co-primary spatial schemes.
A supported effect requires the same sign in both primary designs, an absolute marginal mean difference >= 0.01 AUC in both, and a 95% interval excluding zero in at least one design. A negligible effect requires both primary-design 95% intervals entirely inside [-0.01, +0.01] AUC. Means smaller than 0.01 with wider intervals are uncertain, not negligible. The +/-0.02 interval criterion is *sensitivity only*. Otherwise report inconclusive. Do not make causal claims about pooling or algorithm family solely from paired differences.

## Conditional sensitivity and pilot
A runtime-only pilot may start **after** the full preflight passes and a separate freeze commit records the specification. Pilot scope: outer fold 1 of grid4_s2026_b0; record tuning/fitting runtime only; do not calculate outer-fold metrics or save pilot outer predictions. Buffered 10-km designs are conditional on pilot feasibility (<=60 minutes per design); otherwise record the omission as a deviation. The 23 fixed-parameter schemes are sensitivity analyses, not replacements for the co-primary schemes. The nominal kilometre scale depends on validating the original projected CRS; if not confirmed, label distance assumptions as unresolved.

## Reporting, audit trail and manuscript
Archive per-record OOF predictions, tuning traces, runtime, model failures, coverage, software versions, and any deviations. Avoid leakage through spatial folds or cross-fold parameter reuse. Retain historical manuscript conclusions until new analysis completes. Report pre-specified comparisons separately from exploratory follow-ups. A full successful GitHub preflight and distinct freeze SHA are prerequisites to any formal fitting.

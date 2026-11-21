# v6.24 recovery: restore v6.21 scope and resolve v6.22 protocol deviations

**Status: protocol reconciliation / NOT a completed nested experiment / NOT submission ready.** Date: 2026-11-21.

## Decision

Return to the v6.21 full research scope (MaxEnt, hierarchical Bayesian, two spatial partition designs, pooled-versus-taxon AUC caveat, rare-taxon decisions, calibration and 23-setting robustness), while preserving v6.23's matched 10-km-buffered linear GLM, quadratic GLM and fixed XGBoost comparison as an exploratory component, not the entire manuscript. Never overwrite the v6.21 artifacts.

## Deviations requiring disclosure

| Registered intention | v6.23 implementation | Remedy |
|---|---|---|
| A1 nested spatial XGBoost tuning | Fixed depth-2, 100-tree XGBoost; `nested_tuning=false` | Implement a feasible inner spatial validation and full five-fold outer OOF before any tuned superiority claim |
| Equal-taxon relative log-loss selection | No tuning; previous pilot selected per taxon | Freeze global selector and explicit handling of unestimable taxa before fitting |
| C1–C4 matched comparators | Only fixed XGBoost / linear GLM / quadratic GLM | Restore prespecified comparator matrix and matching features, exclusions, and outer prediction keys |
| Minimum meaningful effect 0.01 | Six-decimal winner framing for a 0.000619 log-loss difference | Use predeclared materiality and uncertainty rule; classify current XGB versus quadratic GLM as practically similar |
| Full v6.21 scientific breadth | Narrowed to one partition / three models | Restore archived analyses with their actual protocols and explicit comparability limits |
| Random-seed robustness | Single seed 622 | Prespecified multi-seed repetitions with exact package versions and paired outcome contrasts |
| Deviation record | Not included in submission candidate | Keep this register in code and final supplement |

## Hard feasibility gate

The original four-inner-group KMeans design **cannot** be executed unchanged. `docs/V622_C1_RARE_CLASS_FEASIBILITY_FINDING.md` documents four zero-training-positive inner cells in CI [#44217152367](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44217152367). For taxon `nsw22` outer fold 5, only one training positive exists, so any exhaustive nonempty inner cross-validation must create a zero-positive training split. **Do not rerun the unchanged design or silently drop invalid splits.**

Before fitting, freeze a revised inner geographic scheme and a separately labeled fallback for that cell; specify the tuning population, spatial separation, objective, fallback hyperparameters, and reporting of excluded/unestimable tuning scores. A seven-taxon selector must never be labeled eight-taxon equal-weight. The new design is a documented post-plan amendment, not retrospective compliance.

## Statistical interpretation corrections

The frozen v6.23 log-loss values are XGBoost 0.186905, quadratic GLM 0.187524, linear GLM 0.204591; Brier scores 0.055984, 0.054642, 0.062737, respectively. An independent rerun reported XGBoost 0.186176 at seed 622 and six-seed log loss 0.1854–0.1866, potentially reflecting dependency/version differences; this has **not yet been reproduced and environment-attributed in CI**. The nominal 0.000619 difference versus quadratic GLM is smaller than the reported seed spread and well below the registered materiality threshold of 0.01. Do not assert XGBoost superiority. Quadratic GLM has lower Brier in the frozen comparison; AUC discrepancies deserve a full table and paired uncertainty assessment.

## Release gates

1. Archive v6.21 original result inventory, figure list, model definitions, and provenance.
2. Freeze revised inner spatial feasibility and rare-class exception **before** tuning; verify every outer/taxon cell without fitting.
3. Execute prespecified pilot, then full nested OOF with global selection and pinned dependencies.
4. Re-run matched C1–C4 arms, paired metrics, spatially defensible uncertainty and seed sensitivity.
5. Integrate the historical models without presenting unmatched protocols as algorithm-only comparisons.
6. Source-verify NSW survey, `disdat` license, predictor units, original citations; clear data redistribution.
7. Produce full EMS-length manuscript, real figures, references, supplementary files and explicit protocol deviations; check journal instructions.

**Do not promote the previous `V623_Submission_Candidate_Package.zip` as submission-ready.**

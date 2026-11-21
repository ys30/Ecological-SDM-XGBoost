# v6.24 evidence inventory

Historical studies use different training sets, feature encodings, kernels or estimators. Preserve their original evaluation populations when importing tables.

| Analysis | Actions evidence | Scope |
| --- | --- | --- |
| NSW source identity | [41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532) | Exact binary-label and site reconciliation against disdat |
| Historical MaxEnt, XGBoost and Bayesian comparison | [41288740752](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41288740752) | Independent verification of 138 archived rank metrics for 23 eligible taxon–fold cells |
| Bayesian and GLM across 5, 10 and 20 km buffers | [41917257355](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41917257355) | Paired evaluation with four model/response variants |
| Alternative spatial partition | [41872107223](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41872107223) | Paired Bayesian and GLM evaluation |
| Calibration | [41772352167](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772352167) | Historical Bayesian calibration audit |
| Rare-taxon false-positive budget | [42108171425](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42108171425) | Archived OOF retrieval audit |
| Earlier tuned/pooled XGBoost sensitivity | [43123540439](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43123540439) | Exploratory tuning-by-sharing results with two partitions and R/Python fixed-control sensitivity on ems-v622-experiments |
| Earlier GLM response specification | [43273382524](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43273382524) | Paired two-design C2b inference |
| Earlier tuned glmnet response specification | [43667019171](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43667019171), [43669427024](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43669427024) | Diagnostic two-design fits and proper-score audit |
| v6.24 buffered eligibility | [44908397667](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44908397667) | 640 cells checked, five invalid training splits and five outer-fold/taxon fallbacks |

The historical 23 paired taxon–fold cells and the 23 distinct spatial sensitivity schemes referenced by the prospective protocol answer different questions. Recovery of the 23-scheme set requires the original scheme list and matching per-scheme output or a versioned rerun. The named v6.21 freeze object 0c159b9 was not present in the publicly reachable branch history checked here. The historical AUC/AP cell audit cannot substitute for the scheme inventory.

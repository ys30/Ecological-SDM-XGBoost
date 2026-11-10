"""EMS v6.22 C1 inner tuning orchestrator.

No outer test rows are accepted here. The caller must pass outer-training data
and its train-only plot-to-inner-fold assignment. The fit callback owns the
model implementation and must not access held-out outer data.
"""
import numpy as np
import pandas as pd
from .xgb_candidate_sampler import generate_candidates
from .xgb_selection_objective import select_candidate_score, choose_candidate
from .fit_retry_policy import run_with_retry

ENV = ("rainann", "soildepth", "tempann", "topo")

def _features(train, validation, variant, taxa):
    if variant not in ("species", "pooled"):
        raise ValueError("C1 supports species and pooled variants only")
    a = train.loc[:, ENV].to_numpy(dtype=float)
    b = validation.loc[:, ENV].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite environmental predictors; train-only preprocessing required")
    if variant == "pooled":
        a = np.column_stack((a, (train["Species"].to_numpy()[:,None] == taxa[None,:]).astype(float)))
        b = np.column_stack((b, (validation["Species"].to_numpy()[:,None] == taxa[None,:]).astype(float)))
    return a, b

def run_c1_inner_tuning(outer_train, inner_plot_to_fold, fit_predict,
                        retry_predict, variant="species", candidates=None):
    """Return tuning choice and diagnostics; never score outer test observations.

    fit_predict(config, train_x, train_y, validation_x, outer_fold_id, taxon)
    retry_predict has the same signature. Caller must implement actual XGBoost
    training and the preregistered retry policy.
    """
    required = {"plot","Species","pres.abs",*ENV}
    if not required <= set(outer_train):
        raise ValueError(f"missing columns: {sorted(required-set(outer_train))}")
    if outer_train.empty or outer_train["plot"].isna().any():
        raise ValueError("empty data or missing plot identifiers")
    if set(inner_plot_to_fold) != set(outer_train["plot"]):
        raise ValueError("inner assignments must cover exactly outer-training plots")
    if len(set(inner_plot_to_fold.values())) != 4:
        raise ValueError("expected four inner folds")
    if not outer_train["pres.abs"].isin([0,1]).all():
        raise ValueError("binary outcome required")
    taxa = np.array(sorted(outer_train["Species"].unique()), dtype=object)
    folds = sorted(set(inner_plot_to_fold.values()))
    if candidates is None:
        candidates = generate_candidates()
    results = []
    details = []
    for candidate in candidates:
        cells = []
        for fold in folds:
            validation = outer_train.loc[outer_train["plot"].map(inner_plot_to_fold) == fold]
            training = outer_train.loc[outer_train["plot"].map(inner_plot_to_fold) != fold]
            if validation.empty or training.empty:
                raise ValueError("empty inner partition")
            if set(validation["plot"]) & set(training["plot"]):
                raise ValueError("plot leakage")
            for taxon in taxa:
                va = validation.loc[validation["Species"] == taxon]
                tr_taxon = training.loc[training["Species"] == taxon]
                if va.empty or tr_taxon.empty:
                    raise ValueError("taxon absent from train or validation: explicit coverage decision required")
                fit_train = tr_taxon if variant == "species" else training
                a, b = _features(fit_train, va, variant, taxa)
                y = fit_train["pres.abs"].to_numpy()
                y_taxon = tr_taxon["pres.abs"].to_numpy()
                y_val = va["pres.abs"].to_numpy()
                # For pooled models, fitting uses all taxa, but class-support
                # fallback and null skill MUST use this taxon's train labels.
                def primary(_x, _y, _v):
                    return fit_predict(candidate, a, y, b, fold, str(taxon))
                def retry(_x, _y, _v):
                    return retry_predict(candidate, a, y, b, fold, str(taxon))
                outcome = run_with_retry(y_taxon, y_val,
                    np.empty((len(y_taxon),0)), np.empty((len(y_val),0)),
                    primary, retry, fitting_support_labels=y if variant == "pooled" else None)
                cells.append({"taxon":str(taxon),"inner_fold":fold,
                    "relative_log_loss_skill":outcome.relative_log_loss_skill})
                details.append({"candidate_id":candidate["candidate_id"],
                    "taxon":str(taxon),"inner_fold":fold,
                    "status":outcome.status,"fallback":outcome.fallback,
                    "attempt_count":outcome.attempt_count,
                    "relative_log_loss_skill":outcome.relative_log_loss_skill,
                    "first_error":outcome.first_error,"retry_error":outcome.retry_error})
        summary = select_candidate_score(cells,expected_taxa=taxa)
        results.append({"candidate_id":candidate["candidate_id"],"score":summary["score"],
                        "cells":summary["cells"],"taxa":summary["taxa"]})
    winner = choose_candidate(results)
    return {"selected_candidate":next(c for c in candidates if c["candidate_id"]==winner["candidate_id"]),
            "candidate_scores":results,"cell_diagnostics":details}

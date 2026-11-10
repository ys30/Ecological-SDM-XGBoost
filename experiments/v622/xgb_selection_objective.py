"""C1 inner-validation selection objective, independent of any model library.

Each (taxon, inner fold) yields a relative log-loss skill computed with that
taxon's inner-training prevalence. The preregistered score weights taxa
equally, then averages their evaluable inner-fold scores.
"""
from collections import defaultdict
import math

def select_candidate_score(rows, expected_taxa=None):
    """Return equal-taxon mean score and coverage diagnostics.

    rows: iterable of dicts with taxon, inner_fold, relative_log_loss_skill.
    Each candidate is scored separately. Reject duplicate taxon/fold cells.
    Missing validation cells are excluded with explicit coverage reporting.
    """
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        taxon = str(row["taxon"])
        fold = int(row["inner_fold"])
        key = (taxon, fold)
        if key in seen:
            raise ValueError("duplicate taxon/inner_fold evaluation cell")
        seen.add(key)
        value = float(row["relative_log_loss_skill"])
        if not math.isfinite(value):
            raise ValueError("nonfinite relative log-loss skill")
        groups[taxon].append(value)
    if not groups:
        raise ValueError("no evaluable inner cells")
    taxa = set(groups)
    if expected_taxa is not None:
        expected = set(map(str, expected_taxa))
        if taxa != expected:
            raise ValueError(f"taxon coverage mismatch: missing={sorted(expected-taxa)} extra={sorted(taxa-expected)}")
    per_taxon = {taxon: sum(values)/len(values) for taxon, values in sorted(groups.items())}
    score = sum(per_taxon.values())/len(per_taxon)
    return {"score": score, "taxa": len(per_taxon), "cells": len(seen),
            "folds_per_taxon": {taxon: len(values) for taxon, values in sorted(groups.items())},
            "per_taxon": per_taxon}

def choose_candidate(candidate_scores):
    """Maximize preregistered skill, break exact ties by smallest candidate_id."""
    if not candidate_scores:
        raise ValueError("empty candidate results")
    for row in candidate_scores:
        if not math.isfinite(float(row["score"])):
            raise ValueError("nonfinite candidate score")
    return min(candidate_scores, key=lambda row: (-float(row["score"]), int(row["candidate_id"])))

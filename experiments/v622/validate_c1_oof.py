"""Validate complete nested spatial OOF prediction tables before any scientific scoring."""
import numpy as np
import pandas as pd

def validate_oof(frame, reference, assignment, expected_folds=5):
    required={"plot","Species","outer_fold","pres.abs","pred","fallback","candidate_id"}
    if not required <= set(frame):
        raise ValueError(f"OOF missing columns: {sorted(required-set(frame))}")
    if not {"plot","Species","pres.abs"} <= set(reference):
        raise ValueError("reference missing plot, Species or outcome")
    if not {"plot","fold"} <= set(assignment) or not assignment["plot"].is_unique:
        raise ValueError("invalid outer fold assignment")
    if assignment["fold"].nunique()!=expected_folds:
        raise ValueError("unexpected outer fold count")
    if len(frame)!=len(reference) or frame[["plot","Species"]].duplicated().any():
        raise ValueError("OOF coverage or uniqueness mismatch")
    keys=["plot","Species"]
    expected=reference[keys+["pres.abs"]].merge(
        assignment[["plot","fold"]],on="plot",validate="many_to_one")
    if len(expected)!=len(reference):
        raise ValueError("reference plots missing assignments")
    merged=frame.merge(expected,on=keys,how="outer",validate="one_to_one",
                       indicator=True,suffixes=("_oof","_ref"))
    if not (merged["_merge"]=="both").all():
        raise ValueError("OOF keys differ from reference")
    if not (merged["pres.abs_oof"]==merged["pres.abs_ref"]).all():
        raise ValueError("OOF outcome mismatch")
    if not (merged["outer_fold"]==merged["fold"]).all():
        raise ValueError("OOF fold mismatch")
    if not np.isfinite(frame["pred"].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite predictions")
    if not frame["pred"].between(0,1).all():
        raise ValueError("prediction outside [0,1]")
    if not frame["fallback"].isin([0,1]).all():
        raise ValueError("invalid fallback flags")
    if frame["candidate_id"].isna().any():
        raise ValueError("missing selected candidate ID")
    if frame.groupby("outer_fold")["candidate_id"].nunique().max()!=1:
        raise ValueError("candidate selection differs within outer fold")
    return {"rows":len(frame),"folds":int(frame["outer_fold"].nunique()),
            "fallback_rows":int(frame["fallback"].sum())}

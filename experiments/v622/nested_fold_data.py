"""EMS v6.22 fold-local nested CV data preparation (no fitting/scoring).

Outer and inner fold membership is plot-based. Inner clusters must be generated
only from outer-training plots. This module does not generate clusters, tune
models, or inspect held-out labels.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .fold_preprocessing import FoldEncoder

@dataclass
class PreparedFold:
    inner_train_x: pd.DataFrame
    inner_validation_x: pd.DataFrame
    outer_test_x: pd.DataFrame
    inner_train_rows: tuple
    inner_validation_rows: tuple
    outer_test_rows: tuple
    columns: tuple

def prepare_nested_fold(frame, outer_test_plots, inner_validation_plots,
                        numeric=("rainann","soildepth","tempann","topo"),
                        categorical=("Species",)):
    if frame["plot"].isna().any():
        raise ValueError("plot identifiers cannot be missing")
    outer_test = set(outer_test_plots)
    inner_validation = set(inner_validation_plots)
    if not outer_test or not inner_validation:
        raise ValueError("outer test and inner validation must both be nonempty")
    if outer_test & inner_validation:
        raise ValueError("inner validation intersects outer held-out plots")
    all_plots = set(frame["plot"])
    if not outer_test <= all_plots or not inner_validation <= all_plots:
        raise ValueError("fold contains plots absent from data")
    outer_mask = frame["plot"].isin(outer_test)
    inner_val_mask = frame["plot"].isin(inner_validation)
    train_mask = ~(outer_mask | inner_val_mask)
    if not train_mask.any():
        raise ValueError("empty inner training partition")
    train = frame.loc[train_mask]
    validation = frame.loc[inner_val_mask]
    test = frame.loc[outer_mask]
    encoder = FoldEncoder(tuple(numeric),tuple(categorical))
    train_x = encoder.fit_transform(train)
    validation_x = encoder.transform(validation)
    test_x = encoder.transform(test)
    assert list(train_x.columns)==list(validation_x.columns)==list(test_x.columns)
    return PreparedFold(train_x,validation_x,test_x,
        tuple(train.index),tuple(validation.index),tuple(test.index),
        tuple(train_x.columns))

def validate_plot_folds(frame, outer_test_plots, inner_plot_to_fold,
                        expected_inner_folds=4):
    """Reject missing/overlapping inner assignments and any outer-test contamination."""
    outer = set(outer_test_plots)
    training = set(frame["plot"]) - outer
    if set(inner_plot_to_fold) != training:
        raise ValueError("inner fold map must cover exactly outer-training plots")
    if any(pd.isna(v) for v in inner_plot_to_fold.values()):
        raise ValueError("missing inner fold label")
    labels = set(inner_plot_to_fold.values())
    if len(labels) != expected_inner_folds:
        raise ValueError("incorrect number of inner folds")
    if any(not any(f==v for f in inner_plot_to_fold.values()) for v in labels):
        raise ValueError("empty inner fold")
    return True

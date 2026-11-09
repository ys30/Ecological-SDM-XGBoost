"""Leakage-safe tabular preprocessing for EMS v6.22.

All learned categories and imputation values come from the training partition.
The returned matrices share identical columns. Unknown categorical values are
encoded as all-zero indicators, never added as new held-out columns.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FoldEncoder:
    numeric: tuple = ("rainann", "soildepth", "tempann", "topo")
    categorical: tuple = ("Species",)
    def fit(self, train):
        self.medians_ = {}
        self.levels_ = {}
        for c in self.numeric:
            if c not in train:
                raise KeyError(c)
            v = pd.to_numeric(train[c], errors="coerce")
            median = v.median()
            if not np.isfinite(median):
                raise ValueError("no finite training values for "+c)
            self.medians_[c] = float(median)
        for c in self.categorical:
            if c not in train:
                raise KeyError(c)
            self.levels_[c] = tuple(sorted(str(x) for x in train[c].dropna().unique()))
        self.columns_ = list(self.numeric) + [
            f"{c}={v}" for c in self.categorical for v in self.levels_[c]
        ]
        return self
    def transform(self, frame):
        if not hasattr(self, "columns_"):
            raise RuntimeError("fit on training data before transform")
        data = {}
        for c in self.numeric:
            data[c] = pd.to_numeric(frame[c], errors="coerce").fillna(self.medians_[c]).to_numpy(dtype=float)
        for c in self.categorical:
            values = frame[c].astype("string")
            for level in self.levels_[c]:
                data[f"{c}={level}"] = values.eq(level).fillna(False).to_numpy(dtype=float)
        out = pd.DataFrame(data, index=frame.index)
        return out.loc[:,self.columns_]
    def fit_transform(self, train):
        return self.fit(train).transform(train)

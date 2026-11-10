"""XGBoost backend for EMS v6.22 C1 inner runner.

The runner calls this adapter once per taxon. Cache successful pooled fits
within a candidate/fold so the same pooled model is trained once and used
for all taxa. Per-taxon fits are never shared across taxa.

A retry doubles boosting rounds; the XGBoost tree booster has no separate
optimizer choice. This interpretation must be approved before study freeze.
"""
import hashlib
import json
import numpy as np

class XGBoostC1Backend:
    def __init__(self, variant, seed=100):
        if variant not in ("species", "pooled"):
            raise ValueError("invalid variant")
        self.variant = variant
        self.seed = int(seed)
        self._cache = {}
        self.fit_count = 0

    def _key(self, config, x, y, fold, taxon, retry):
        # Include training bytes to prevent accidental cross-split reuse.
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        digest = hashlib.sha256(x.tobytes()+y.tobytes()+repr(x.shape).encode()).hexdigest()
        return (json.dumps(config, sort_keys=True, allow_nan=False),int(fold),
                None if self.variant=="pooled" else str(taxon),
                digest, bool(retry))

    def _predict(self, config, train_x, train_y, validation_x, fold, taxon, retry):
        from xgboost import XGBClassifier
        x = np.asarray(train_x,dtype=float)
        y = np.asarray(train_y,dtype=int)
        v = np.asarray(validation_x,dtype=float)
        if x.ndim!=2 or v.ndim!=2 or x.shape[1]!=v.shape[1]:
            raise ValueError("invalid XGBoost matrices")
        if len(x)!=len(y) or len(v)==0 or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("invalid XGBoost input rows or features")
        if set(np.unique(y)) != {0,1}:
            raise ValueError("XGBoost fitting requires both training classes")
        key=self._key(config,x,y,fold,taxon,retry)
        model=self._cache.get(key)
        if model is None:
            model=XGBClassifier(
                n_estimators=int(config["n_estimators"])*(2 if retry else 1),
                max_depth=int(config["max_depth"]),
                learning_rate=float(config["learning_rate"]),
                min_child_weight=float(config["min_child_weight"]),
                subsample=float(config["subsample"]),
                colsample_bytree=float(config["colsample_bytree"]),
                objective="binary:logistic",tree_method="hist",
                random_state=self.seed, n_jobs=2,eval_metric="logloss",
            )
            model.fit(x,y)
            self._cache[key]=model
            self.fit_count+=1
        return np.asarray(model.predict_proba(v)[:,1],dtype=float)

    def primary(self, config, train_x, train_y, validation_x, fold, taxon):
        return self._predict(config,train_x,train_y,validation_x,fold,taxon,False)

    def retry(self, config, train_x, train_y, validation_x, fold, taxon):
        return self._predict(config,train_x,train_y,validation_x,fold,taxon,True)

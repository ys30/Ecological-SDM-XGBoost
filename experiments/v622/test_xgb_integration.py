"""Small real-XGBoost integration smoke; synthetic data only.

Exercises all C1 layers together without outer-test data or scientific metrics.
"""
import unittest
import numpy as np
import pandas as pd
from experiments.v622.xgb_backend import XGBoostC1Backend
from experiments.v622.xgb_inner_runner import run_c1_inner_tuning

class TestC1Integration(unittest.TestCase):
    def test_real_fit_pooled_once_per_inner_fold(self):
        rows=[]
        for plot in range(40):
            for taxon in ("a","b"):
                rows.append({"plot":plot,"Species":taxon,
                    "pres.abs":int((plot+int(taxon=="b"))%3==0),
                    "rainann":float(plot)/40,"soildepth":float(plot%7),
                    "tempann":float(plot%9),"topo":float(plot%5)})
        data=pd.DataFrame(rows)
        folds={plot:plot%4 for plot in range(40)}
        config={"candidate_id":1,"n_estimators":5,"max_depth":2,
                "learning_rate":.1,"min_child_weight":.5,
                "subsample":1.,"colsample_bytree":1.}
        backend=XGBoostC1Backend("pooled")
        result=run_c1_inner_tuning(data,folds,backend.primary,backend.retry,
                                   variant="pooled",candidates=[config])
        self.assertEqual(backend.fit_count,4)
        self.assertEqual(len(result["cell_diagnostics"]),8)
        self.assertTrue(all(d["status"]=="fit_success" for d in result["cell_diagnostics"]))
        self.assertTrue(np.isfinite(result["candidate_scores"][0]["score"]))

    def test_pooled_predicts_rare_taxon_when_shared_fit_supported(self):
        rows=[]
        for plot in range(40):
            for taxon in ("a","b"):
                rows.append({"plot":plot,"Species":taxon,
                    "pres.abs":int(taxon=="a" and plot%3==0),
                    "rainann":float(plot)/40,"soildepth":float(plot%7),
                    "tempann":float(plot%9),"topo":float(plot%5)})
        data=pd.DataFrame(rows)
        folds={plot:plot%4 for plot in range(40)}
        config={"candidate_id":1,"n_estimators":5,"max_depth":2,
                "learning_rate":.1,"min_child_weight":.5,
                "subsample":1.,"colsample_bytree":1.}
        pooled=XGBoostC1Backend("pooled")
        result=run_c1_inner_tuning(data,folds,pooled.primary,pooled.retry,
                                   variant="pooled",candidates=[config])
        rare=[x for x in result["cell_diagnostics"] if x["taxon"]=="b"]
        self.assertEqual(len(rare),4)
        self.assertTrue(all(not x["fallback"] for x in rare))
        self.assertEqual(pooled.fit_count,4)
        species=XGBoostC1Backend("species")
        other=run_c1_inner_tuning(data,folds,species.primary,species.retry,
                                   variant="species",candidates=[config])
        self.assertTrue(all(x["fallback"] for x in other["cell_diagnostics"]
                            if x["taxon"]=="b"))

if __name__=="__main__":
    unittest.main()

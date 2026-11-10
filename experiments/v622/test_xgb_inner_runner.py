import unittest
import numpy as np
import pandas as pd
from experiments.v622.xgb_inner_runner import run_c1_inner_tuning

class TestC1InnerRunner(unittest.TestCase):
    def setUp(self):
        plots = list(range(40))
        rows = []
        for plot in plots:
            for taxon in ("a","b"):
                rows.append({"plot":plot,"Species":taxon,"pres.abs":int(plot%3==0),
                    "rainann":float(plot),"soildepth":2.,"tempann":20.,"topo":1.})
        self.data = pd.DataFrame(rows)
        self.folds = {p:p%4 for p in plots}
        self.candidates = [{"candidate_id":1}]

    def test_species_and_pooled_no_outer_data(self):
        calls=[]
        def fit(config,x,y,v,fold,taxon):
            calls.append((len(x),len(y),len(v),fold,taxon))
            return np.full(len(v),.33)
        for variant in ("species","pooled"):
            result=run_c1_inner_tuning(self.data,self.folds,fit,fit,variant,self.candidates)
            self.assertEqual(result["selected_candidate"]["candidate_id"],1)
            self.assertEqual(len(result["cell_diagnostics"]),8)
            self.assertTrue(all(not x["fallback"] for x in result["cell_diagnostics"]))
        self.assertTrue(any(c[0]==30 for c in calls))
        self.assertTrue(any(c[0]==60 for c in calls))

    def test_plot_coverage_rejected(self):
        with self.assertRaises(ValueError):
            run_c1_inner_tuning(self.data,{0:0},lambda *a:None,lambda *a:None,
                                candidates=self.candidates)

    def test_failure_fallback_is_logged(self):
        def fail(*args): raise RuntimeError("intentional")
        result=run_c1_inner_tuning(self.data,self.folds,fail,fail,
                                    candidates=self.candidates)
        self.assertTrue(all(x["fallback"] for x in result["cell_diagnostics"]))
        self.assertTrue(all(x["attempt_count"]==2 for x in result["cell_diagnostics"]))

if __name__=="__main__":
    unittest.main()

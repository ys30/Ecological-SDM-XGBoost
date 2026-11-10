import unittest
import numpy as np
from experiments.v622.xgb_backend import XGBoostC1Backend

class TestBackend(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(80,dtype=float).reshape(20,4)/80
        self.y=np.array([0,1]*10)
        self.v=np.ones((3,4),dtype=float)*.2
        self.config={"candidate_id":1,"n_estimators":5,"max_depth":2,
            "learning_rate":.1,"min_child_weight":.5,"subsample":1.,
            "colsample_bytree":1.}

    def test_pooled_fit_shared_across_taxa(self):
        b=XGBoostC1Backend("pooled")
        p=b.primary(self.config,self.x,self.y,self.v,0,"a")
        q=b.primary(self.config,self.x,self.y,self.v,0,"b")
        self.assertEqual(b.fit_count,1)
        np.testing.assert_allclose(p,q)
        self.assertTrue(np.isfinite(p).all())

    def test_species_fit_is_not_shared(self):
        b=XGBoostC1Backend("species")
        b.primary(self.config,self.x,self.y,self.v,0,"a")
        b.primary(self.config,self.x,self.y,self.v,0,"b")
        self.assertEqual(b.fit_count,2)

    def test_retry_uses_distinct_model(self):
        b=XGBoostC1Backend("pooled")
        b.primary(self.config,self.x,self.y,self.v,0,"a")
        b.retry(self.config,self.x,self.y,self.v,0,"a")
        self.assertEqual(b.fit_count,2)

if __name__=="__main__":
    unittest.main()

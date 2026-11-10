import unittest
import numpy as np
from experiments.v622.glm_adapter import run_glm_inner
class TestGLMAdapter(unittest.TestCase):
    def test_real_fit_without_outer_metrics(self):
        x=np.arange(40,dtype=float).reshape(20,2)/40
        y=np.array([0,1]*10)
        v=np.array([[.3,.4],[.6,.7]])
        r=run_glm_inner(y,[0,1],x,v)
        self.assertIn(r.status,("fit_success","retry_success"))
        self.assertFalse(r.fallback)
        self.assertTrue(np.isfinite(r.predictions).all())
        self.assertEqual(r.predictions.shape,(2,))
    def test_rare_taxon_never_fits(self):
        x=np.arange(12,dtype=float).reshape(6,2)
        r=run_glm_inner([0,0,0,0,0,1],[0,1],x,np.zeros((2,2)))
        self.assertEqual(r.status,"insufficient_positives")
        self.assertEqual(r.attempt_count,0)
        np.testing.assert_allclose(r.predictions,[1/6]*2)
    def test_feature_dimensions_guarded(self):
        x=np.arange(40,dtype=float).reshape(20,2)/40
        y=np.array([0,1]*10)
        r=run_glm_inner(y,[0,1],x,np.zeros((2,3)))
        self.assertEqual(r.status,"fit_failure_fallback")
        self.assertEqual(r.attempt_count,2)
        self.assertIn("incompatible design matrices",r.first_error)
if __name__=="__main__":
    unittest.main()

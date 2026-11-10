import unittest
import numpy as np
from experiments.v622.inner_taxon_dispatch import run_inner_taxon
class TestDispatch(unittest.TestCase):
    def test_fallback_never_fits_and_uses_raw_training_prevalence(self):
        def must_not_fit(*args):
            raise AssertionError("fit must not run")
        r=run_inner_taxon([0,0,0,1],[0,1],np.zeros((4,2)),np.zeros((2,2)),must_not_fit)
        self.assertTrue(r.fallback)
        np.testing.assert_array_equal(r.predictions,[.25,.25])
        self.assertFalse(r.validation_auc_defined)
        self.assertTrue(np.isfinite(r.relative_log_loss_skill))
    def test_fitted_case(self):
        r=run_inner_taxon([1]*5+[0]*5,[0,1],np.zeros((10,2)),np.zeros((2,2)),
                          lambda x,y,v:np.array([.1,.8]))
        self.assertFalse(r.fallback)
        self.assertTrue(r.validation_auc_defined)
        self.assertEqual(r.reason,"fitted")
    def test_one_class_validation_still_scores(self):
        r=run_inner_taxon([1]*5+[0]*5,[0,0],np.zeros((10,1)),np.zeros((2,1)),
                          lambda x,y,v:np.array([.1,.2]))
        self.assertFalse(r.validation_auc_defined)
        self.assertTrue(np.isfinite(r.relative_log_loss_skill))
    def test_invalid_predictions_fail_closed(self):
        with self.assertRaises(ValueError):
            run_inner_taxon([1]*5,[0,1],np.zeros((5,1)),np.zeros((2,1)),
                            lambda x,y,v:[.1])
    def test_threshold_locked(self):
        with self.assertRaises(ValueError):
            run_inner_taxon([0,1],[0],np.zeros((2,1)),np.zeros((1,1)),
                            lambda x,y,v:[.2],minimum_positives=2)
if __name__=="__main__":
    unittest.main()

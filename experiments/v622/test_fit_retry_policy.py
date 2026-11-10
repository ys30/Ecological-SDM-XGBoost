import unittest
import numpy as np
from experiments.v622.fit_retry_policy import run_with_retry
class TestRetry(unittest.TestCase):
    def args(self):
        return ([1]*5+[0]*5,[0,1],np.zeros((10,2)),np.zeros((2,2)))
    def test_primary_success_skips_retry(self):
        def retry(*a):raise AssertionError("retry must not run")
        r=run_with_retry(*self.args(),lambda *a:[.2,.8],retry)
        self.assertEqual((r.status,r.attempt_count,r.fallback),("fit_success",1,False))
    def test_retry_success_records_first_failure(self):
        def fail(*a):raise RuntimeError("first")
        r=run_with_retry(*self.args(),fail,lambda *a:[.2,.8])
        self.assertEqual((r.status,r.attempt_count,r.fallback),("retry_success",2,False))
        self.assertIn("first",r.first_error)
    def test_double_failure_fallback_and_auc_excluded(self):
        def fail(*a):raise RuntimeError("failure")
        r=run_with_retry(*self.args(),fail,fail)
        self.assertEqual((r.status,r.attempt_count,r.fallback),("fit_failure_fallback",2,True))
        self.assertFalse(r.auc_eligible)
        np.testing.assert_allclose(r.predictions,[.5,.5])
        self.assertTrue(np.isfinite(r.relative_log_loss_skill))
    def test_insufficient_positives_skips_all_fits(self):
        def fail(*a):raise AssertionError("must not fit")
        r=run_with_retry([0,0,1],[0,1],np.zeros((3,1)),np.zeros((2,1)),fail,fail)
        self.assertEqual((r.status,r.attempt_count),("insufficient_positives",0))
    def test_invalid_primary_predictions_trigger_retry(self):
        r=run_with_retry(*self.args(),lambda *a:[float("nan"),.2],lambda *a:[.1,.9])
        self.assertEqual(r.status,"retry_success")
if __name__=="__main__":
    unittest.main()

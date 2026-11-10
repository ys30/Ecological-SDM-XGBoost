import unittest
import numpy as np
from experiments.v622.class_support_policy import (
    fallback_required,fallback_predict,null_prevalence,log_loss,
    relative_log_loss_skill,auc_eligible)
class TestPolicy(unittest.TestCase):
    def test_training_only_fallback(self):
        tr=[0,0,0,1]
        self.assertTrue(fallback_required(tr))
        self.assertFalse(fallback_required([1]*5))
        np.testing.assert_array_equal(fallback_predict(tr,3),[.25]*3)
        self.assertEqual(null_prevalence(tr),.3)
    def test_zero_positive_training_is_safe(self):
        tr=[0,0,0]
        self.assertTrue(fallback_required(tr))
        self.assertEqual(fallback_predict(tr,2).tolist(),[0,0])
        self.assertGreater(null_prevalence(tr),0)
        score=relative_log_loss_skill(tr,[0,0],[0,0])
        self.assertTrue(np.isfinite(score))
    def test_single_class_validation_has_log_loss_but_no_auc(self):
        self.assertTrue(np.isfinite(relative_log_loss_skill([0,1,1,0],[0,0],[.2,.3])))
        self.assertFalse(auc_eligible([0,0]))
        self.assertFalse(auc_eligible([0,1],model_a_fallback=True))
        self.assertTrue(auc_eligible([0,1]))
    def test_null_is_training_only(self):
        tr=[0,0,0,1]
        self.assertAlmostEqual(relative_log_loss_skill(tr,[1,1],[.3,.3]),0)
    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            fallback_required([0,2])
        with self.assertRaises(ValueError):
            log_loss([0,1],[.1,float("nan")])
if __name__=="__main__":
    unittest.main()

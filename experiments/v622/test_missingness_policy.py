import unittest
import numpy as np
import pandas as pd
from experiments.v622.design_matrix import FoldDesign
class TestMissingnessPolicy(unittest.TestCase):
    def data(self):
        return pd.DataFrame({"rainann":[1.,2.,3.],"soildepth":[1.,2.,3.],
                             "tempann":[3.,4.,5.],"topo":[5.,6.,7.]})
    def test_strict_default_rejects_training_missing(self):
        d=self.data()
        d.loc[0,"rainann"]=np.nan
        with self.assertRaisesRegex(ValueError,"training predictor"):
            FoldDesign("glm_quad_rt").fit(d)
    def test_strict_default_rejects_validation_missing(self):
        d=self.data()
        model=FoldDesign("glm_quad_rt").fit(d)
        d.loc[0,"topo"]=np.nan
        with self.assertRaisesRegex(ValueError,"during transform"):
            model.transform(d)
    def test_explicit_train_median_only(self):
        d=self.data()
        d.loc[0,"rainann"]=np.nan
        model=FoldDesign("glm_quad_rt",missing_policy="train_median").fit(d)
        self.assertEqual(model.median_["rainann"],2.5)
        self.assertEqual(model.transform(d).loc[0,"rainann"],2.5)
    def test_reject_unregistered_policy(self):
        with self.assertRaises(ValueError):
            FoldDesign("glm_lin",missing_policy="silent")
if __name__=="__main__":
    unittest.main()

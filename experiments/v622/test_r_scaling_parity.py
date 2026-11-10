import unittest
import numpy as np
import pandas as pd
from experiments.v622.design_matrix import FoldDesign

class TestRScalingParity(unittest.TestCase):
    def test_r_sample_sd_and_squared_z(self):
        d=pd.DataFrame({"rainann":[1.,2.,4.,8.],"soildepth":[2.,3.,5.,7.],
                        "tempann":[10.,12.,14.,18.],"topo":[1.,3.,4.,9.]})
        design=FoldDesign("glm_quad_rt",standardize=True).fit(d)
        x=design.transform(d)
        for c in ("rainann","soildepth","tempann","topo"):
            expected=(d[c]-d[c].mean())/d[c].std(ddof=1)
            np.testing.assert_allclose(x[c],expected,atol=1e-12)
            self.assertAlmostEqual(design.scale_[c],float(d[c].std(ddof=1)))
        np.testing.assert_allclose(x["rainann_sq"],x["rainann"]**2)
        np.testing.assert_allclose(x["tempann_sq"],x["tempann"]**2)
    def test_holdout_moments_do_not_change(self):
        d=pd.DataFrame({"rainann":[1.,2.,3.],"soildepth":[2.,3.,4.],
                        "tempann":[10.,20.,30.],"topo":[4.,5.,6.]})
        a=FoldDesign("glm_quad_rt",standardize=True).fit(d)
        v=d.iloc[[0]].copy()
        v["rainann"]=1e12
        a.transform(v)
        self.assertAlmostEqual(a.scale_["rainann"],1.)
    def test_constant_column_finite(self):
        d=pd.DataFrame({"rainann":[1.,1.,1.],"soildepth":[1.,2.,3.],
                        "tempann":[2.,3.,4.],"topo":[3.,4.,5.]})
        a=FoldDesign("glm_quad_rt",standardize=True).fit(d)
        self.assertEqual(a.scale_["rainann"],1.)
        self.assertTrue(np.isfinite(a.transform(d).to_numpy()).all())
if __name__=="__main__":
    unittest.main()

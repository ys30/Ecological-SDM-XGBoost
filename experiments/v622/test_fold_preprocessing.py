import unittest
import numpy as np
import pandas as pd
from experiments.v622.fold_preprocessing import FoldEncoder

class TestFoldEncoder(unittest.TestCase):
    def test_no_holdout_category_or_median_leakage(self):
        tr = pd.DataFrame({"rainann":[1.,np.nan,3.],"soildepth":[1.,2.,3.],
          "tempann":[4.,5.,6.],"topo":[7.,8.,9.],"Species":["a","b","a"]})
        te = pd.DataFrame({"rainann":[np.nan,100000.],"soildepth":[2.,2.],
          "tempann":[4.,4.],"topo":[7.,7.],"Species":["unseen","a"]})
        enc = FoldEncoder().fit(tr)
        x=enc.transform(tr);y=enc.transform(te)
        self.assertEqual(list(x.columns),list(y.columns))
        self.assertEqual(enc.medians_["rainann"],2.)
        self.assertEqual(y.iloc[0]["rainann"],2.)
        self.assertEqual(y.iloc[0]["Species=a"],0.)
        self.assertEqual(y.iloc[0]["Species=b"],0.)
        self.assertNotIn("Species=unseen",y.columns)
        self.assertEqual(y.iloc[1]["Species=a"],1.)
    def test_no_finite_numeric_training_values(self):
        tr=pd.DataFrame({"rainann":[np.nan],"soildepth":[1.],"tempann":[1.],"topo":[1.],"Species":["a"]})
        with self.assertRaises(ValueError):
            FoldEncoder().fit(tr)
    def test_missing_fit(self):
        with self.assertRaises(RuntimeError):
            FoldEncoder().transform(pd.DataFrame())
if __name__=="__main__":
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from experiments.v622.nested_fold_data import prepare_nested_fold,validate_plot_folds

def sample():
    return pd.DataFrame({
      "plot":[1,1,2,2,3,3,4,4,5,5,6,6],
      "Species":["a","b"]*6,
      "rainann":[1.,2.,np.nan,4.,5.,6.,7.,8.,9.,10.,999.,999.],
      "soildepth":[1.]*12,"tempann":[2.]*12,"topo":[3.]*12,
    })
class NestedFoldTests(unittest.TestCase):
    def test_disjoint_and_no_outer_leakage(self):
        d=sample()
        p=prepare_nested_fold(d,{6},{5})
        self.assertEqual(len(p.inner_train_rows),8)
        self.assertEqual(len(p.inner_validation_rows),2)
        self.assertEqual(len(p.outer_test_rows),2)
        self.assertEqual(tuple(p.inner_train_x.columns),p.columns)
        self.assertEqual(tuple(p.outer_test_x.columns),p.columns)
        self.assertEqual(p.inner_train_x.loc[2,"rainann"],5.)
        self.assertEqual(p.outer_test_x.loc[10,"rainann"],999.)
    def test_outer_inner_overlap_rejected(self):
        with self.assertRaises(ValueError):
            prepare_nested_fold(sample(),{6},{6})
    def test_missing_plot_rejected(self):
        with self.assertRaises(ValueError):
            prepare_nested_fold(sample(),{7},{5})
    def test_inner_assignments(self):
        d=sample()
        self.assertTrue(validate_plot_folds(d,{6},{1:0,2:1,3:2,4:3,5:0}))
        with self.assertRaises(ValueError):
            validate_plot_folds(d,{6},{1:0,2:1,3:2,4:3,5:0,6:1})
        with self.assertRaises(ValueError):
            validate_plot_folds(d,{6},{1:0,2:1,3:2,4:3})
if __name__=="__main__":
    unittest.main()

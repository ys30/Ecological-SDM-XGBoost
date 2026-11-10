import unittest
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.nested_fold_data import validate_plot_folds

class TestSpatialInner(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({"plot":list(range(1,25)),
            "easting":[i*100. for i in range(24)],
            "northing":[(i%5)*50. for i in range(24)]})
    def test_determinism_and_outer_exclusion(self):
        d=self.sample()
        a=spatial_inner_folds(d,{23,24},n_clusters=8,n_folds=4)
        b=spatial_inner_folds(d.sample(frac=1,random_state=99),{23,24},n_clusters=8,n_folds=4)
        pd.testing.assert_frame_equal(a,b)
        self.assertFalse(set(a["plot"])&{23,24})
        self.assertEqual(a["inner_fold"].nunique(),4)
        self.assertTrue(validate_plot_folds(d,{23,24},dict(zip(a["plot"],a["inner_fold"]))))
        self.assertEqual(a.groupby("cluster")["inner_fold"].nunique().max(),1)
    def test_coordinate_inconsistency(self):
        d=self.sample()
        d=pd.concat([d,d.iloc[[0]].assign(easting=9999.)])
        with self.assertRaises(ValueError):
            spatial_inner_folds(d,{24},n_clusters=8)
if __name__=="__main__":
    unittest.main()

"""Synthetic spatial-fold rare-taxon regression; no outer scoring."""
import unittest
import numpy as np
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.design_matrix import prepare_design_fold
from experiments.v622.glm_adapter import run_glm_inner

class RareSpatialFallback(unittest.TestCase):
    def test_four_positives_fallback_on_all_inner_folds(self):
        rng=np.random.default_rng(622)
        plot=np.arange(160)
        frame=pd.DataFrame({
            "plot":plot,"easting":(plot%20)*1000.,
            "northing":(plot//20)*1000.,
            "rainann":rng.normal(100,20,160),
            "soildepth":rng.normal(30,5,160),
            "tempann":rng.normal(20,3,160),
            "topo":rng.normal(0,1,160),
            "Species":["rare"]*160,"pres.abs":np.zeros(160,dtype=int)})
        outer=set(plot[::5])
        folds=spatial_inner_folds(frame,outer,outer_fold=1)
        frame.loc[frame["plot"].isin(folds["plot"].iloc[:4]),"pres.abs"]=1
        for k in range(4):
            val_plots=set(folds.loc[folds["inner_fold"]==k,"plot"])
            train,val,_,_=prepare_design_fold(
                frame,outer,val_plots,"glm_quad_rt",standardize=True)
            trmask=~frame["plot"].isin(outer|val_plots)
            vamask=frame["plot"].isin(val_plots)
            ytr=frame.loc[trmask,"pres.abs"].to_numpy()
            yval=frame.loc[vamask,"pres.abs"].to_numpy()
            result=run_glm_inner(ytr,yval,train.to_numpy(),val.to_numpy())
            self.assertEqual(result.status,"insufficient_positives")
            self.assertEqual(result.attempt_count,0)
            self.assertTrue(result.fallback)
            self.assertFalse(result.auc_eligible)
            np.testing.assert_allclose(result.predictions,ytr.mean())
            self.assertTrue(np.isfinite(result.relative_log_loss_skill))

if __name__=="__main__":
    unittest.main()

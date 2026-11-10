import unittest
import pandas as pd
from experiments.v622.feature_contract import predictor_columns,prepare_for_model
class TestFeatures(unittest.TestCase):
    def test_taxon_only_in_pooled(self):
        for model in ["glm_lin","glm_quad_rt","xgb_species_tuned","maxent_tuned","glmnet_quad_rt"]:
            n,c=predictor_columns(model)
            self.assertEqual(c,())
            self.assertEqual(len(n),4)
        for model in ["glm_shared_lin","xgb_pooled_tuned"]:
            n,c=predictor_columns(model)
            self.assertEqual(c,("Species",))
            self.assertEqual(len(n),4)
    def test_full_pooled_extra(self):
        n,c=predictor_columns("xgb_pooled_full_tuned")
        self.assertEqual(n,("rainann","soildepth","tempann","topo","soilfert","disturb"))
        self.assertEqual(c,("Species",))
    def test_no_target_leakage(self):
        for model in ["glm_lin","glm_shared_lin","xgb_pooled_full_tuned"]:
            n,c=predictor_columns(model)
            self.assertFalse(set(n+c)&{"pres.abs","plot","id","easting","northing"})
    def test_real_fold_data(self):
        rows=[]
        for plot in range(1,7):
            for species in ["nsw18","nsw19"]:
                rows.append(dict(plot=plot,Species=species,**{"pres.abs":int(plot%2==0)},
                    rainann=100+plot,soildepth=10+plot,tempann=20+plot,topo=plot,
                    soilfert=plot%3,disturb=plot%2))
        d=pd.DataFrame(rows)
        for model in ["glm_lin","glm_shared_lin","xgb_pooled_full_tuned"]:
            p=prepare_for_model(d,{6},{5},model)
            self.assertEqual(len(p.inner_train_x)+len(p.inner_validation_x)+len(p.outer_test_x),len(d))
            self.assertFalse(set(p.columns)&{"pres.abs","plot","id"})
            self.assertEqual(any(c.startswith("Species=") for c in p.columns),model!="glm_lin")
if __name__=="__main__":
    unittest.main()

"""Explicit feature contracts from v6.22 model matrix; never infer predictors from all columns."""
BASE_NUMERIC=("rainann","soildepth","tempann","topo")
FULL_EXTRA=("soilfert","disturb")
TAXON_COLUMN="Species"
RESPONSE_COLUMN="pres.abs"
FORBIDDEN={"plot","id","pres.abs","long","lat","easting","northing"}

def predictor_columns(model_id):
    pooled={"glm_shared_lin","glm_shared_quad_rt","xgb_pooled_fixed","xgb_pooled_tuned","xgb_pooled_full_tuned"}
    species={"glm_lin","glm_quad_rt","glm_quad_env","glmm_lin","glmm_quad_rt",
             "bayes_lin","bayes_quad","glmnet_lin","glmnet_quad_env","glmnet_quad_rt",
             "maxent_tuned","maxent_env","xgb_species_fixed","xgb_species_tuned"}
    if model_id not in pooled|species:
        raise ValueError(f"unrecognized model_id: {model_id}")
    numeric=BASE_NUMERIC+(FULL_EXTRA if model_id=="xgb_pooled_full_tuned" else ())
    categorical=(TAXON_COLUMN,) if model_id in pooled else ()
    assert not FORBIDDEN.intersection(numeric+categorical)
    return numeric,categorical

def prepare_for_model(frame,outer_test_plots,inner_validation_plots,model_id):
    from .nested_fold_data import prepare_nested_fold
    numeric,categorical=predictor_columns(model_id)
    return prepare_nested_fold(frame,outer_test_plots,inner_validation_plots,
                               numeric=numeric,categorical=categorical)

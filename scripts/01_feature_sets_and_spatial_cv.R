# Part 1: Feature Subset Experiments
## The goal is to compare model performance using:
### All features
### Only environmental or only geographic features

#Step-by-step for Feature Subset Experiment (using XGBoost):
# --- Define feature subsets ---
env_features <- c("rainann", "tempann", "soildepth",
                  grep("^soilfert[1-5]$", names(train_train), value = TRUE),
                  grep("^disturb[1-4]$", names(train_train), value = TRUE),
                  "topo")
env_features
geo_features1 <- c("easting", "northing")
geo_features2 <- c("long", "lat")
all_features1 <- setdiff(
  c(env_features, geo_features1, grep("^Species", names(train_train), value = TRUE)),
  "Species"
)
all_features1
all_features2 <- setdiff(
  c(env_features, geo_features2, grep("^Species", names(train_train), value = TRUE)),
  "Species"
)
all_features2

# --- Helper function to train and evaluate XGBoost model ---
run_xgb_eval <- function(feature_set) {
  # Select training and validation feature sets
  X_train <- train_train[, feature_set]
  X_val <- train_val[, feature_set]
  
  # Drop non-numeric columns (e.g., characters or factors)
  X_train <- X_train[, sapply(X_train, is.numeric)]
  X_val <- X_val[, sapply(X_val, is.numeric)]
  
  # Convert to matrix
  X_train <- as.matrix(X_train)
  X_val <- as.matrix(X_val)
  
  y_train <- train_train$pres.abs
  y_val <- train_val$pres.abs
  
  # Fit XGBoost
  model <- xgboost(data = X_train, label = y_train, 
                   objective = "binary:logistic",
                   eval_metric = "logloss", 
                   nrounds = 100, max_depth = 4, eta = 0.1,
                   verbose = 0)
  
  # Predict and clip
  probs <- predict(model, X_val)
  probs_clipped <- pmin(pmax(probs, 1e-15), 1 - 1e-15)
  
  # Metrics
  loss <- logLoss(y_val, probs_clipped)
  auc_val <- pROC::auc(pROC::roc(y_val, probs_clipped))
  
  return(c(LogLoss = round(loss, 4), AUC = round(auc_val, 4)))
}


# --- Run experiments ---
result_all1 <- run_xgb_eval(all_features1)
result_all2 <- run_xgb_eval(all_features2)
result_env <- run_xgb_eval(env_features)
result_geo1 <- run_xgb_eval(geo_features1)
result_geo2 <- run_xgb_eval(geo_features2)

# --- Summarize results ---
subset_results <- rbind(
  All_Features1 = result_all1,
  All_Features2 = result_all2,
  Environmental_Only = result_env,
  Geographic_Only1 = result_geo1,
  Geographic_Only2 = result_geo2
)

print(subset_results)


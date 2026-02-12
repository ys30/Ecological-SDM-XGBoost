library(xgboost)
library(pROC)
library(dplyr)

# 0. Assume best parameters are given
best_params <- list(
  eta       = 0.01,
  max_depth = 4
)


# 1. Prepare full training data
X_train_full <- as.matrix(train[, all_features2])
y_train_full <- train$pres.abs  # Binary labels (0/1)


# 2. Cross-validation to find best nrounds
cv <- xgb.cv(
  data                  = X_train_full,
  label                 = y_train_full,
  nrounds               = 2000,  # Upper limit
  eta                   = best_params$eta,
  max_depth             = best_params$max_depth,
  nfold                 = 5,
  objective             = "binary:logistic",
  eval_metric           = "logloss",
  early_stopping_rounds = 30,
  verbose               = 0
)
best_nrounds <- cv$best_iteration
cat("Best number of boosting rounds:", best_nrounds, "\n")


# 3. Train final model on full data
xgb_model_final <- xgboost(
  data        = X_train_full,
  label       = y_train_full,
  nrounds     = best_nrounds,
  eta         = best_params$eta,
  max_depth   = best_params$max_depth,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  objective   = "binary:logistic",
  eval_metric = "logloss",
  verbose     = 0
)


# 4. Feature importance plot
importance_tbl <- xgb.importance(model = xgb_model_final)
xgb.plot.importance(importance_tbl, top_n = 20,
                    main = "Top 20 Important Features")


# 5. Predict on test set
X_test_full <- as.matrix(test[, all_features2])
test_probs  <- predict(xgb_model_final, X_test_full)
test_probs_clipped <- pmin(pmax(test_probs, 1e-15), 1 - 1e-15)

submission <- data.frame(
  id   = test$id,
  pred = test_probs_clipped
)
write.csv(submission, "submission_xgb_full.csv", row.names = FALSE)

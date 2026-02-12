# Repeat the Full Evaluation Pipeline with best model

# Step 1: Prepare matrices
X_train_xgb <- as.matrix(train_train[, all_features2])
X_val_xgb   <- as.matrix(train_val[, all_features2])
y_train_xgb <- train_train$pres.abs
y_val_xgb   <- train_val$pres.abs

# Step 2: Train XGBoost model using best tuned parameters
xgb_model_best <- xgboost(data = X_train_xgb,
                          label = y_train_xgb,
                          objective = "binary:logistic",
                          eval_metric = "logloss",
                          nrounds = best_params$nrounds,
                          max_depth = best_params$max_depth,
                          eta = best_params$eta,
                          verbose = 0)

# Step 3: Predict and log-loss
xgb_probs_best <- predict(xgb_model_best, X_val_xgb)
xgb_probs_best_clipped <- pmin(pmax(xgb_probs_best, 1e-15), 1 - 1e-15)
xgb_log_loss_best <- logLoss(y_val_xgb, xgb_probs_best_clipped)

# Step 4: ROC and AUC
xgb_roc_best <- pROC::roc(y_val_xgb, xgb_probs_best_clipped)
xgb_auc_best <- pROC::auc(xgb_roc_best)

# Plot
# plot(roc_log, col = "blue", lwd = 2, main = "ROC Curve - Validation Set")
# lines(roc_rf, col = "darkgreen", lwd = 2)
# lines(xgb_roc, col = "green", lwd = 2)
# lines(xgb_roc_best, col = "red", lwd = 2)
# legend("bottomright", 
#        legend = c("Logistic Regression", "Random Forest", "XGBoost", "Tuned XGBoost"),
#        col = c("blue", "darkgreen", "green", "red"), lwd = 2)

plot(xgb_roc_best, col = "blue", lwd = 2, main = "ROC Curve - Tuned XGBoost")

# Step 5: Classification Metrics
threshold <- 0.5
xgb_class_best <- ifelse(xgb_probs_best_clipped > threshold, 1, 0)
xgb_cm_best <- confusionMatrix(factor(xgb_class_best), factor(y_val_xgb), positive = "1")

# Step 6: Print results
cat("Tuned XGBoost Log Loss:", round(xgb_log_loss_best, 4), "\n")
cat("Tuned XGBoost AUC:", round(xgb_auc_best, 4), "\n")
print(xgb_cm_best)

# Step 7: Feature Importance
xgb.importance(model = xgb_model_best) %>%
  xgb.plot.importance(top_n = 10)

# Step 8: Predict on Test Set
X_test_xgb <- as.matrix(test[, all_features2])
xgb_test_probs_best <- predict(xgb_model_best, X_test_xgb)
xgb_test_probs_best_clipped <- pmin(pmax(xgb_test_probs_best, 1e-15), 1 - 1e-15)

submission_best <- data.frame(id = test$id, pred = xgb_test_probs_best_clipped)
write.csv(submission_best, "submission_xgb_tuned.csv", row.names = FALSE)


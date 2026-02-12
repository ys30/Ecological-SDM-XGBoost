# Part 2: Spatial Cross-Validation
# For ecological modeling, nearby plots are often correlated, so spatial CV is important.
# Simple Simulation of Spatial Block CV:
# Create 5 spatial blocks using longitude
library(dplyr)
library(caret)

# Bin by longitude (simplified blocking)
train$spatial_block <- ntile(train$long, 5)

# Placeholder to collect fold scores
logloss_vec <- c()
auc_vec <- c()

for (fold in 1:5) {
  train_fold <- train %>% filter(spatial_block != fold)
  val_fold <- train %>% filter(spatial_block == fold)
  
  X_train <- as.matrix(train_fold[, all_features2])
  X_val <- as.matrix(val_fold[, all_features2])
  y_train <- train_fold$pres.abs
  y_val <- val_fold$pres.abs
  
  model <- xgboost(data = X_train, label = y_train, objective = "binary:logistic",
                   eval_metric = "logloss", nrounds = 100, max_depth = 4, eta = 0.1, verbose = 0)
  
  probs <- predict(model, X_val)
  probs_clipped <- pmin(pmax(probs, 1e-15), 1 - 1e-15)
  
  logloss_vec <- c(logloss_vec, logLoss(y_val, probs_clipped))
  auc_vec <- c(auc_vec, pROC::auc(roc(y_val, probs_clipped)))
}

cat("Mean Log-Loss (Spatial CV):", round(mean(logloss_vec), 4), "\n")
cat("Mean AUC (Spatial CV):", round(mean(auc_vec), 4), "\n")


### explain: We conducted spatial 5-fold cross-validation by partitioning plots based on longitude. The mean log-loss of 0.1869 and AUC of 0.8906 indicate strong performance, even under geographically explicit validation. This confirms the model's robustness and generalization to new spatial regions, an important consideration in ecological prediction.

# Part3. Model tuning
# Step 1: Define Tuning Grid
library(caret)

# Tuning grid for XGBoost
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150, 200, 250),
  max_depth = c(3, 5, 7, 9),
  eta = c(0.01, 0.05, 0.1, 0.2),
  gamma = 0, colsample_bytree = 1,
  min_child_weight = 1, subsample = 1
)

# Step 2: Cross-Validation Setup (use logLoss and AUC)
# Custom summary function to use logLoss and AUC
loglossSummary <- function(data, lev = NULL, model = NULL) {
  # Check that "Yes" column exists
  if (!"Yes" %in% colnames(data)) {
    stop("Probability column 'Yes' not found. Check factor labels and classProbs = TRUE.")
  }
  
  probs <- data[, "Yes"]
  y_true <- ifelse(data$obs == "Yes", 1, 0)
  
  logloss <- Metrics::logLoss(y_true, probs)
  auc_val <- pROC::auc(pROC::roc(y_true, probs))
  
  out <- c(logLoss = logloss, AUC = as.numeric(auc_val))
  names(out) <- c("logLoss", "AUC")
  return(out)
}


ctrl <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE,             # VERY important for logLoss
  summaryFunction = loglossSummary,
  savePredictions = TRUE
)


# Step 3: Train Tuned XGBoost Model
# Data setup
X <- train_train[, all_features2]
X <- X[, sapply(X, is.numeric)]  # ensure numeric
X <- as.matrix(X)
# Convert response to factor with valid level names
y <- factor(train_train$pres.abs, levels = c(0, 1), labels = c("No", "Yes"))

# Train with caret
set.seed(123)
xgb_tuned <- train(
  x = X, y = y,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "logLoss"
)



# Step 4: Review Results
print(xgb_tuned)
plot(xgb_tuned)
best_params <- xgb_tuned$bestTune
print(best_params)

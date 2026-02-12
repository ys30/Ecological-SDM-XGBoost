setwd("C:/Users/tangl/Dropbox/250606_SDM")

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(Metrics)
library(corrplot)
library(reshape2)
library(dplyr)

# install.packages("corrplot")

# Step 1: Load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Step 2: Exploratory Data Analysis (EDA)
## general information
str(train)
str(test)

#Table 1 and 2
summary(train)
summary(test)

# Calculate standard deviation for all numeric columns
numeric_cols <- sapply(train, is.numeric)
sd_values <- sapply(train[, numeric_cols], sd, na.rm = TRUE)
# View the result
sd_values


# Table 3 - Frequency and Correlation of Species Co-occurrence Variables
# Count frequency of each species
species_freq <- train %>%
  group_by(Species) %>%
  summarise(
    Presence_Count = sum(pres.abs == 1),
    Total_Count = n(),
    Correlation_with_Target = round(cor(as.numeric(pres.abs), as.numeric(Species == unique(Species)[1])), 3) # temporary placeholder
  ) %>%
  ungroup()
# Update correlation values for each species
species_freq$Correlation_with_Target <- sapply(unique(train$Species), function(sp) {
  cor(as.numeric(train$Species == sp), train$pres.abs)
})
# Round correlation for display
species_freq$Correlation_with_Target <- round(species_freq$Correlation_with_Target, 3)
# Arrange by descending correlation
table3 <- species_freq %>%
  arrange(desc(Correlation_with_Target))
# View table
print(table3)


## Fig.1 Target distribution
ggplot(train, aes(x = factor(pres.abs))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Presence/Absence Distribution", x = "Presence (1) / Absence (0)", y = "Count")

## Fig.2 Species distribution
# Convert pres.abs to factor for grouped fill
train1 <- train
train1$pres.abs <- factor(train$pres.abs, labels = c("Absent", "Present"))
ggplot(train1, aes(x = Species, fill = pres.abs)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("darkred", "forestgreen")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Species Frequency by Presence/Absence",
       x = "Species", y = "Count", fill = "Presence Status")


# Fig.3 histogram 
# Select key environmental variables
env_vars <- train %>%
  dplyr::select(tempann, rainann, soildepth, topo,soilfert,disturb )
str(env_vars)
# Convert to long format for facet plotting
env_long <- env_vars %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")
str(env_long)
# Plot using density or histogram (use one or the other depending on clarity)
ggplot(env_long, aes(x = Value)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(~Variable, scales = "free", ncol = 2) +
  theme_minimal(base_size = 13) +
  labs(title = "Distribution of Key Environmental Predictors",
       x = "Value", y = "Frequency")


# Fig 4. Box plot 
# Select spatial variables
spatial_vars <- train %>%
  dplyr::select(long, lat, easting, northing)
# Reshape to long format
spatial_long <- spatial_vars %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")
# Create boxplots
# Faceted box plots
ggplot(spatial_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", color = "black", outlier.color = "red") +
  facet_wrap(~Variable, scales = "free", ncol = 2) +
  theme_minimal(base_size = 13) +
  labs(title = "Box Plots of Spatial Variables",
       x = NULL, y = "Value")


# Fig 5. Correlation
## Correlation matrix (numeric features only)
num_vars <- train %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-id, -plot, -pres.abs)  # exclude ID columns and target

corr_matrix <- cor(num_vars)
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8, title = "Correlation Matrix")


# Step 3: Preprocessing
## One-hot encode Species
train$Species <- as.factor(train$Species)
test$Species <- as.factor(test$Species)

train$disturb <- as.factor(train$disturb)
test$disturb <- as.factor(test$disturb)

train$soilfert <- as.factor(train$soilfert)
test$soilfert <- as.factor(test$soilfert)

str(train)
str(test)

train <- cbind(train, model.matrix(~Species - 1, train))
test <- cbind(test, model.matrix(~Species - 1, test))

train <- cbind(train, model.matrix(~disturb - 1, train))
test <- cbind(test, model.matrix(~disturb - 1, test))

train <- cbind(train, model.matrix(~soilfert - 1, train))
test <- cbind(test, model.matrix(~soilfert - 1, test))

str(train)
str(test)

summary(train)
summary(test)

## Feature set
features <- c("rainann", "soildepth", "tempann", "topo", "easting", "northing", 
              grep("^Species", names(train), value = TRUE),
              grep("^disturb", names(train), value = TRUE),
              grep("^soilfert", names(train), value = TRUE))
features
features <-  features[-7]
features
features <-  features[-15]
features
features <-  features[-19]
features

## Align test features with train
missing_cols <- setdiff(features, names(test))
missing_cols
test[missing_cols] <- 0  # add missing dummies with 0

# Step 4: Train-test split
set.seed(42)
split <- createDataPartition(train$pres.abs, p = 0.8, list = FALSE)
train_train <- train[split, ]
train_val <- train[-split, ]

# Step 5: Modeling
## Logistic Regression
log_model <- glm(pres.abs ~ ., data = train_train[, c("pres.abs", features)], family = binomial)
summary(log_model)

#Logistic Regression Coefficients and Interpretation
# Display coefficients
summary(log_model)$coefficients
# p-present probabality, odds = p/(1-p), log(p/(1-p))
# Interpretation: The coefficient represents the log odds change of pres.abs = 1 (i.e., presence) 
## for a one-unit increase in the predictor, holding others constant.

#To convert log-odds to odds ratios:
exp(coef(log_model))  # Odds Ratios
## Example:   If tempann has a coefficient of 0.003:
###  Log-odds ??? by 0.003 per 1-unit increase in tempann
### Odds Ratio = exp(0.003) ???  1.003 ??? an 103% increase in odds of presence per 
### unit of temperature.
### odds ratio = p/(1-p) = 1.003

#prediction
log_probs <- predict(log_model, newdata = train_val[, features], type = "response")
log_loss_val <- logLoss(train_val$pres.abs, log_probs)


## Random Forest
# Use matrix interface to avoid formula parsing errors
rf_model <- randomForest(x = train_train[, features], 
                         y = as.factor(train_train$pres.abs), 
                         ntree = 100)
rf_model

# random Forest Variable Importance
# Get importance scores
importance_values <- importance(rf_model)
# Plot variable importance
varImpPlot(rf_model, main = "Random Forest Variable Importance")
## Interpretation:  Higher MeanDecreaseGini means the variable splits nodes better, 
### increasing model purity. 
## Important variables contribute more to reducing classification error.

# Predict on Validation Set 
rf_probs <- predict(rf_model, newdata = train_val[, features], type = "prob")[, 2]

# Clip probabilities
rf_probs_clipped <- pmin(pmax(rf_probs, 1e-15), 1 - 1e-15) # Clip probabilities to avoid log(0)
# Compute log-loss again
rf_log_loss <- logLoss(train_val$pres.abs, rf_probs_clipped)


# Step 6: Evaluate
#6.1 Log Loss
cat("Logistic Regression Log Loss:", round(log_loss_val, 4), "\n")
cat("Random Forest Log Loss:", round(rf_log_loss, 4), "\n")

# 6.2. Plot ROC Curves
library(pROC)

# ROC for Logistic Regression
roc_log <- roc(train_val$pres.abs, log_probs)
# ROC for Random Forest
roc_rf <- roc(train_val$pres.abs, rf_probs_clipped)

# Plot ROC
plot(roc_log, col = "blue", lwd = 2, main = "ROC Curve - Validation Set")
lines(roc_rf, col = "darkgreen", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "Random Forest"),
       col = c("blue", "darkgreen"), lwd = 2)

# Print AUC
cat("Logistic Regression AUC:", pROC::auc(roc_log), "\n")
cat("Random Forest AUC:", pROC::auc(roc_rf), "\n")

# 6.3. Precision, Recall, F1 Score
# Choose a threshold (e.g., 0.5 or best F1 threshold you found earlier):
library(caret)

# Choose threshold (use tuned one or default 0.5)
threshold <- 0.5

# Binarize predictions
log_pred_class <- ifelse(log_probs > threshold, 1, 0)
rf_pred_class <- ifelse(rf_probs_clipped > threshold, 1, 0)

# Confusion matrices
log_cm <- confusionMatrix(factor(log_pred_class), factor(train_val$pres.abs), positive = "1")
rf_cm <- confusionMatrix(factor(rf_pred_class), factor(train_val$pres.abs), positive = "1")

# Print results
print(log_cm)
print(rf_cm)


# Step 7: Predict on test set
final_probs <- predict(rf_model, newdata = test[, features], type = "prob")[, 2]
final_probs_clipped <- pmin(pmax(final_probs, 1e-15), 1 - 1e-15)
final_probs_clipped

# Choose threshold (use tuned one or default 0.5)
#threshold <- 0.5

# Binarize predictions
# rf_pred_rest_class <- ifelse(final_probs_clipped > threshold, 1, 0)
# rf_pred_rest_class

# Step 8: Create submission file
submission <- data.frame(id = test$id, pred = final_probs_clipped)
str(submission)
write.csv(submission, "submission.csv", row.names = FALSE)



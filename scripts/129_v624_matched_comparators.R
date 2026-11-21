# Matched fixed logistic and presence-background model fits.
# The MaxEnt scores are used only for discrimination.
suppressPackageStartupMessages({
  library(glmnet)
  library(maxnet)
})
output <- "results/v624_comparators"
features <- c("rainann", "soildepth", "tempann", "topo")
data <- read.csv("data/train.csv")
membership <- read.csv(file.path(output, "membership.csv"))
taxa <- sort(unique(data$Species))
predictions <- list()
coverage <- list()
emit <- function(test, probability, fold, arm) {
  stopifnot(length(probability) == nrow(test), all(is.finite(probability)))
  data.frame(id = test$id, plot = test$plot, taxon = test$Species,
             outer = fold, y = test$pres.abs, arm = arm, p = probability)
}
emit_coverage <- function(fold, taxon, count, status) {
  data.frame(outer = fold, taxon = taxon, positive_train = count, status = status)
}
ridge <- function(x, y, test, penalty = rep(1, ncol(x))) {
  fit <- glmnet(x, y, family = "binomial", alpha = 0, lambda = .01,
                standardize = FALSE, penalty.factor = penalty)
  as.numeric(predict(fit, test, s = .01, type = "response"))
}
for (fold in 1:5) {
  fold_rows <- membership[membership$outer == fold, ]
  train <- data[data$plot %in% fold_rows$plot[fold_rows$role == "train"], ]
  test <- data[data$plot %in% fold_rows$plot[fold_rows$role == "test"], ]
  stopifnot(!length(intersect(train$plot, test$plot)))
  ztrain <- scale(train[, features])
  ztest <- scale(test[, features], center = attr(ztrain, "scaled:center"),
                 scale = attr(ztrain, "scaled:scale"))
  dtrain <- model.matrix(~ 0 + taxon, list(taxon = factor(train$Species, levels = taxa)))
  dtest <- model.matrix(~ 0 + taxon, list(taxon = factor(test$Species, levels = taxa)))
  for (quadratic in c(FALSE, TRUE)) {
    suffix <- if (quadratic) "_quadratic" else "_linear"
    x <- if (quadratic) cbind(ztrain, ztrain[, c(1, 3)]^2) else ztrain
    v <- if (quadratic) cbind(ztest, ztest[, c(1, 3)]^2) else ztest
    full_x <- cbind(dtrain[, -1], x)
    full_v <- cbind(dtest[, -1], v)
    partial_x <- cbind(full_x, do.call(cbind, lapply(seq_along(taxa), function(j) x * dtrain[, j])))
    partial_v <- cbind(full_v, do.call(cbind, lapply(seq_along(taxa), function(j) v * dtest[, j])))
    for (sharing in c("full_slopes", "partial_slopes")) {
      a <- if (sharing == "full_slopes") full_x else partial_x
      b <- if (sharing == "full_slopes") full_v else partial_v
      penalty <- c(rep(0, length(taxa) - 1), rep(1, ncol(x)),
                   rep(4, ncol(a) - ncol(full_x)))
      key <- paste0(sharing, suffix)
      predictions[[length(predictions) + 1L]] <- emit(test, ridge(a, train$pres.abs, b, penalty), fold, key)
    }
    for (taxon in taxa) {
      tr <- train$Species == taxon
      te <- test$Species == taxon
      stopifnot(length(unique(train$pres.abs[tr])) == 2)
      probability <- withCallingHandlers(
        ridge(x[tr, , drop = FALSE], train$pres.abs[tr], v[te, , drop = FALSE]),
        warning = function(w) {
          if (!grepl("fewer than 8 observations", conditionMessage(w), fixed = TRUE)) {
            stop(conditionMessage(w))
          }
          invokeRestart("muffleWarning")
        }
      )
      predictions[[length(predictions) + 1L]] <- emit(test[te, ], probability, fold,
                                                       paste0("no_sharing", suffix))
    }
  }
  for (taxon in taxa) {
    tr <- train[train$Species == taxon, ]
    te <- test[test$Species == taxon, ]
    n_positive <- sum(tr$pres.abs)
    if (n_positive < 10 || sum(tr$pres.abs == 0) < 10) {
      coverage[[length(coverage) + 1L]] <- emit_coverage(fold, taxon, n_positive,
                                                         "insufficient_training_support")
      next
    }
    x <- tr[, features]
    model <- maxnet(tr$pres.abs, x, maxnet.formula(tr$pres.abs, x, classes = "lq"),
                    regmult = 2, addsamplestobackground = FALSE)
    score <- as.numeric(predict(model, te[, features], type = "cloglog"))
    coverage[[length(coverage) + 1L]] <- emit_coverage(fold, taxon, n_positive, "fit")
    predictions[[length(predictions) + 1L]] <- emit(te, score, fold, "maxent_fixed")
  }
}
predictions <- do.call(rbind, predictions)
stopifnot(!anyDuplicated(predictions[, c("id", "arm")]))
counts <- table(predictions$arm)
stopifnot(all(counts[names(counts) != "maxent_fixed"] == nrow(data)))
write.csv(predictions, file.path(output, "predictions.csv"), row.names = FALSE)
write.csv(do.call(rbind, coverage), file.path(output, "maxent_coverage.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output, "r_session.txt"))
print(counts)

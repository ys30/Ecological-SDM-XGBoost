# Reproducible holdout evaluation; independent of interactive R session.
# Run from repository root. This is a NEW benchmark, not a reproduction
# of the historical manuscript's exact tuning configuration.
suppressPackageStartupMessages({
  library(xgboost)
  library(ggplot2)
  library(pROC)
})
set.seed(42)
d <- read.csv("data/train.csv", stringsAsFactors = FALSE)
stopifnot(all(c("Species","disturb","soilfert","pres.abs","plot") %in% names(d)))
stopifnot(!anyNA(d), all(d$pres.abs %in% c(0,1)))
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
# Hold out whole plots to avoid same-plot records appearing in both sets.
plots <- unique(d$plot)
set.seed(42)
holdout_plots <- sample(plots, size = ceiling(.2 * length(plots)))
i_val <- d$plot %in% holdout_plots
stopifnot(!any(d$plot[!i_val] %in% d$plot[i_val]))
# One-hot encode using the complete fixed taxonomy/categories; no label used.
cat_cols <- c("Species","disturb","soilfert")
for (nm in cat_cols) d[[nm]] <- factor(d[[nm]])
num_cols <- c("rainann","soildepth","tempann","topo","long","lat")
x <- model.matrix(~ . - 1, data = d[, c(num_cols,cat_cols), drop = FALSE])
y <- d$pres.abs
stopifnot(length(unique(y[i_val])) == 2L, length(unique(y[!i_val])) == 2L)
# Internal grouped CV selects nrounds, never using holdout labels.
# Fold assignment is at plot level, not row level.
tr <- which(!i_val)
tr_plots <- unique(d$plot[tr])
set.seed(43)
fold_ids <- sample(rep(1:5, length.out = length(tr_plots)))
folds <- lapply(1:5, function(k) which(d$plot[tr] %in% tr_plots[fold_ids == k]))
stopifnot(all(vapply(folds, length, integer(1)) > 0))
params <- list(objective="binary:logistic", eval_metric="logloss",
               eta=.03, max_depth=4, subsample=.8,
               colsample_bytree=.8, nthread=2)
cv <- xgb.cv(params=params, data=xgb.DMatrix(x[tr,,drop=FALSE],label=y[tr]),
             nrounds=500, folds=folds, early_stopping_rounds=25,
             verbose=0)
# XGBoost R releases differ in whether best_iteration is populated.
# Prefer explicit best_iteration; otherwise use the minimum CV test logloss.
best_round <- cv$best_iteration
if (is.null(best_round) || length(best_round) != 1L ||
    !is.finite(best_round) || best_round < 1) {
  ev <- as.data.frame(cv$evaluation_log)
  score_col <- grep("test.*logloss.*mean", names(ev), value = TRUE)
  if (length(score_col) != 1L) {
    stop("Cannot identify test logloss column in xgb.cv evaluation_log: ",
         paste(names(ev), collapse = ", "))
  }
  scores <- as.numeric(ev[[score_col]])
  if (!length(scores) || all(!is.finite(scores))) {
    stop("Cross-validation produced no finite test logloss scores")
  }
  best_round <- which.min(replace(scores, !is.finite(scores), Inf))
  if ("iter" %in% names(ev)) best_round <- as.integer(ev$iter[best_round])
}
best_round <- as.integer(best_round)
stopifnot(length(best_round) == 1L, !is.na(best_round), best_round >= 1L)
cat("Selected boosting rounds:", best_round, "\n")
set.seed(44)
fit <- xgb.train(params=params, data=xgb.DMatrix(x[tr,,drop=FALSE],label=y[tr]),
                 nrounds=best_round, verbose=0)
p <- pmin(pmax(predict(fit,x[i_val,,drop=FALSE]),1e-15),1-1e-15)
truth <- y[i_val]
roc_obj <- pROC::roc(truth,p,levels=c(0,1),direction="<",quiet=TRUE)
roc_auc <- as.numeric(pROC::auc(roc_obj))
logloss <- -mean(truth*log(p)+(1-truth)*log1p(-p))
# Average precision (stepwise area under precision-recall curve).
o <- order(p,decreasing=TRUE)
tp <- cumsum(truth[o] == 1)
precision <- tp/seq_along(tp)
recall <- tp/sum(truth == 1)
ap <- sum(precision[truth[o] == 1])/sum(truth == 1)
pred <- as.integer(p >= .5)
tp_n <- sum(pred==1 & truth==1); fp_n <- sum(pred==1 & truth==0)
tn_n <- sum(pred==0 & truth==0); fn_n <- sum(pred==0 & truth==1)
safe_div <- function(a,b) if(b==0) NA_real_ else a/b
metrics <- data.frame(model="XGBoost_plot_grouped_holdout",train_rows=length(tr),
                      validation_rows=sum(i_val), train_plots=length(tr_plots),
                      validation_plots=length(holdout_plots),
                      best_nrounds=best_round,roc_auc=roc_auc,
                      average_precision=ap,log_loss=logloss,
                      precision=safe_div(tp_n,tp_n+fp_n),
                      recall=safe_div(tp_n,tp_n+fn_n),
                      specificity=safe_div(tn_n,tn_n+fp_n),
                      f1=safe_div(2*tp_n,2*tp_n+fp_n+fn_n),
                      tp=tp_n,fp=fp_n,tn=tn_n,fn=fn_n)
write.csv(metrics,"results/plot_holdout_metrics.csv",row.names=FALSE)
write.csv(data.frame(id=d$id[i_val],plot=d$plot[i_val],
                     observed=truth,predicted_probability=p),
          "results/plot_holdout_predictions.csv",row.names=FALSE)
roc_df <- data.frame(fpr=1-roc_obj$specificities,tpr=roc_obj$sensitivities)
g1 <- ggplot(roc_df,aes(fpr,tpr))+geom_line(color="#2669a0",linewidth=1)+
  geom_abline(linetype=2,color="grey60")+
  coord_equal()+labs(title=sprintf("Plot-held-out ROC | AUC %.3f",roc_auc),
  x="False positive rate",y="True positive rate")+theme_minimal(base_size=13)
ggsave("results/figures/06_plot_holdout_roc.png",g1,width=7,height=6,dpi=160)
pr_df <- data.frame(recall=c(0,recall),precision=c(1,precision))
g2 <- ggplot(pr_df,aes(recall,precision))+geom_step(color="#237b61",linewidth=.8)+
  geom_hline(yintercept=mean(truth),linetype=2,color="grey60")+
  labs(title=sprintf("Plot-held-out precision–recall | AP %.3f",ap),
  x="Recall",y="Precision")+theme_minimal(base_size=13)
ggsave("results/figures/07_plot_holdout_pr.png",g2,width=7,height=6,dpi=160)
imp <- xgb.importance(feature_names=colnames(x),model=fit)
write.csv(imp,"results/plot_holdout_feature_importance.csv",row.names=FALSE)
imp <- head(imp,15)
g3 <- ggplot(imp,aes(x=reorder(Feature,Gain),y=Gain))+
  geom_col(fill="#356f9c")+coord_flip()+
  labs(title="XGBoost gain importance | plot-grouped holdout",
       x=NULL,y="Relative gain")+theme_minimal(base_size=12)
ggsave("results/figures/08_plot_holdout_feature_importance.png",
       g3,width=9,height=7,dpi=160)
print(metrics)

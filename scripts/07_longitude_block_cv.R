# Geographic-block cross-validation for ecological SDM.
# Each fold holds out a contiguous longitude band of unique plots.
# This is a stronger geographic extrapolation stress test than random plot holdout,
# but longitude-only bands are not a full two-dimensional spatial-block design.
suppressPackageStartupMessages({
  library(xgboost)
  library(ggplot2)
  library(pROC)
})
d <- read.csv("data/train.csv", stringsAsFactors=FALSE)
stopifnot(!anyNA(d), all(d$pres.abs %in% c(0,1)))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
# Split at plot level. Use plot median longitude to avoid separating records.
plot_long <- aggregate(long ~ plot, data=d, FUN=median)
plot_long <- plot_long[order(plot_long$long,plot_long$plot),]
plot_long$fold <- pmin(5L, ceiling(seq_len(nrow(plot_long))*5/nrow(plot_long)))
fold <- plot_long$fold[match(d$plot,plot_long$plot)]
stopifnot(!anyNA(fold), length(unique(fold))==5)
# Explicitly remove longitude/latitude from predictors in this geographic
# extrapolation benchmark, to test climate/soil/taxon transfer rather than
# location memorization. Report this distinct feature set transparently.
for (nm in c("Species","disturb","soilfert")) d[[nm]] <- factor(d[[nm]])
features <- c("rainann","soildepth","tempann","topo","Species","disturb","soilfert")
x <- model.matrix(~ . - 1,data=d[,features,drop=FALSE])
y <- d$pres.abs
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,
               max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
safe_div <- function(a,b) if(b>0) a/b else NA_real_
all_pred <- vector("list",5)
all_metrics <- vector("list",5)
for(k in 1:5){
  tr <- which(fold!=k); te <- which(fold==k)
  stopifnot(length(intersect(unique(d$plot[tr]),unique(d$plot[te])))==0)
  # Fixed boosting budget across all folds avoids tuning on the held-out
  # geographic band. This is a benchmark hyperparameter choice, not nested CV.
  set.seed(100+k)
  fit <- xgb.train(params=params,
                   data=xgb.DMatrix(x[tr,,drop=FALSE],label=y[tr]),
                   nrounds=300,verbose=0)
  p <- pmin(pmax(predict(fit,x[te,,drop=FALSE]),1e-15),1-1e-15)
  t <- y[te]
  pred <- as.integer(p>=.5)
  tp <- sum(pred==1 & t==1); fp <- sum(pred==1 & t==0)
  tn <- sum(pred==0 & t==0); fn <- sum(pred==0 & t==1)
  auc <- if(length(unique(t))==2) as.numeric(pROC::auc(
    pROC::roc(t,p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
  ap <- if(sum(t)>0) {
    o <- order(p,decreasing=TRUE)
    sum((cumsum(t[o]==1)/seq_along(o))[t[o]==1])/sum(t)
  } else NA_real_
  all_metrics[[k]] <- data.frame(fold=k,train_rows=length(tr),
    validation_rows=length(te),validation_plots=length(unique(d$plot[te])),
    longitude_min=min(d$long[te]),longitude_max=max(d$long[te]),
    prevalence=mean(t),roc_auc=auc,average_precision=ap,
    log_loss=-mean(t*log(p)+(1-t)*log1p(-p)),
    precision=safe_div(tp,tp+fp),recall=safe_div(tp,tp+fn),
    f1=safe_div(2*tp,2*tp+fp+fn),tp=tp,fp=fp,tn=tn,fn=fn)
  all_pred[[k]] <- data.frame(id=d$id[te],plot=d$plot[te],fold=k,
    observed=t,predicted_probability=p)
}
metrics <- do.call(rbind,all_metrics)
predictions <- do.call(rbind,all_pred)
write.csv(metrics,"results/longitude_block_cv_metrics.csv",row.names=FALSE)
write.csv(predictions,"results/longitude_block_cv_predictions.csv",row.names=FALSE)
summary <- data.frame(folds=5,mean_roc_auc=mean(metrics$roc_auc,na.rm=TRUE),
  sd_roc_auc=sd(metrics$roc_auc,na.rm=TRUE),
  mean_average_precision=mean(metrics$average_precision,na.rm=TRUE),
  mean_log_loss=mean(metrics$log_loss),pooled_prevalence=mean(y),
  n_missing_fold_auc=sum(is.na(metrics$roc_auc)))
write.csv(summary,"results/longitude_block_cv_summary.csv",row.names=FALSE)
g <- ggplot(metrics,aes(x=factor(fold),y=roc_auc))+
  geom_col(fill="#386c9e")+coord_cartesian(ylim=c(0,1))+
  labs(title="Geographic transfer: longitude-band holdout",
       subtitle="Five non-overlapping plot-level longitude bands; coordinates excluded from features",
       x="Held-out geographic band",y="ROC AUC")+theme_minimal(base_size=12)
ggsave("results/figures/09_longitude_block_cv_auc.png",g,width=9,height=5,dpi=160)
g2 <- ggplot(metrics,aes(x=factor(fold),y=average_precision))+
  geom_col(fill="#287a69")+geom_point(aes(y=prevalence),shape=4,size=3)+
  coord_cartesian(ylim=c(0,1))+
  labs(title="Geographic transfer: precision–recall performance",
       subtitle="Bars: average precision; crosses: prevalence baseline",
       x="Held-out geographic band",y="Average precision / prevalence")+
  theme_minimal(base_size=12)
ggsave("results/figures/10_longitude_block_cv_ap.png",g2,width=9,height=5,dpi=160)
print(metrics)
print(summary)

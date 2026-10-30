# Matched-feature evaluation: plot holdout vs longitude-band CV.
# Both protocols use identical environmental + taxon features, XGBoost
# hyperparameters and a fixed 300 boosting rounds (no holdout tuning).
suppressPackageStartupMessages({library(xgboost);library(pROC);library(ggplot2)})
d <- read.csv("data/train.csv",stringsAsFactors=FALSE)
stopifnot(!anyNA(d),all(d$pres.abs %in% c(0,1)))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
for(nm in c("Species","disturb","soilfert")) d[[nm]] <- factor(d[[nm]])
features <- c("rainann","soildepth","tempann","topo","Species","disturb","soilfert")
x <- model.matrix(~ . - 1,d[,features,drop=FALSE])
y <- d$pres.abs
params <- list(objective="binary:logistic",eval_metric="logloss",
               eta=.03,max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
plots <- unique(d$plot)
set.seed(42)
held <- sample(plots,ceiling(.2*length(plots)))
plot_holdout <- ifelse(d$plot %in% held,1L,0L)
pl <- aggregate(long~plot,data=d,FUN=median)
pl <- pl[order(pl$long,pl$plot),]
pl$fold <- pmin(5L,ceiling(seq_len(nrow(pl))*5/nrow(pl)))
geo_fold <- pl$fold[match(d$plot,pl$plot)]
evaluate <- function(protocol,k,tr,te) {
  stopifnot(!length(intersect(unique(d$plot[tr]),unique(d$plot[te]))))
  set.seed(100+k)
  fit <- xgb.train(params=params,
    data=xgb.DMatrix(x[tr,,drop=FALSE],label=y[tr]),nrounds=300,verbose=0)
  p <- pmin(pmax(predict(fit,x[te,,drop=FALSE]),1e-15),1-1e-15)
  t <- y[te]
  roc_auc <- if(length(unique(t))==2) as.numeric(pROC::auc(
    pROC::roc(t,p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
  ap <- if(sum(t)>0) {o <- order(p,decreasing=TRUE)
    sum((cumsum(t[o]==1)/seq_along(o))[t[o]==1])/sum(t)} else NA_real_
  pred <- p>=.5; tp <- sum(pred & t==1); fn <- sum(!pred & t==1)
  data.frame(protocol=protocol,fold=k,train_rows=length(tr),
    validation_rows=length(te),validation_plots=length(unique(d$plot[te])),
    prevalence=mean(t),roc_auc=roc_auc,average_precision=ap,
    log_loss=-mean(t*log(p)+(1-t)*log1p(-p)),
    recall=if(tp+fn>0) tp/(tp+fn) else NA_real_)
}
rows <- list(evaluate("Random plot holdout",1,
  which(plot_holdout==0),which(plot_holdout==1)))
for(k in 1:5) rows[[k+1]] <- evaluate("Longitude-band CV",k,
  which(geo_fold!=k),which(geo_fold==k))
metrics <- do.call(rbind,rows)
write.csv(metrics,"results/matched_protocol_metrics.csv",row.names=FALSE)
summary <- aggregate(cbind(roc_auc,average_precision,log_loss,recall)~protocol,
  data=metrics,FUN=mean)
write.csv(summary,"results/matched_protocol_summary.csv",row.names=FALSE)
plot_data <- rbind(
  data.frame(protocol=metrics$protocol,fold=metrics$fold,
             metric="ROC AUC",value=metrics$roc_auc),
  data.frame(protocol=metrics$protocol,fold=metrics$fold,
             metric="Average precision",value=metrics$average_precision),
  data.frame(protocol=metrics$protocol,fold=metrics$fold,
             metric="Recall at 0.5",value=metrics$recall))
g <- ggplot(plot_data,aes(x=protocol,y=value,color=protocol))+
  geom_point(size=3,position=position_jitter(width=.08,height=0))+
  stat_summary(fun=mean,geom="crossbar",width=.35,color="black")+
  facet_wrap(~metric,scales="free_y")+
  labs(title="Matched-feature validation protocol comparison",
       subtitle="Identical predictors and 300-round XGBoost; five geographic folds vs one plot holdout",
       x=NULL,y="Validation metric")+
  theme_minimal(base_size=12)+theme(axis.text.x=element_text(angle=18,hjust=1),
    legend.position="none")
ggsave("results/figures/11_matched_protocol_comparison.png",g,width=11,height=5,dpi=160)
print(metrics);print(summary)

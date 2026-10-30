# Two-dimensional spatial-block cross-validation (native longitude/latitude grid).
# Blocks are formed without response labels, then assigned deterministically
# to folds. Plot identity is kept intact. This is a spatial robustness stress
# test, not distance-buffered CV or independent external validation.
suppressPackageStartupMessages({library(xgboost);library(pROC);library(ggplot2)})
d <- read.csv("data/train.csv",stringsAsFactors=FALSE)
stopifnot(!anyNA(d),all(d$pres.abs %in% c(0,1)))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
plots <- aggregate(cbind(long,lat)~plot,data=d,FUN=median)
# Quantile grid ensures roughly occupied cells, but individual cells vary.
# 4x4 quantile cells grouped into five folds with a deterministic shuffled
# greedy assignment balancing plot counts; response labels never used.
qx <- unique(quantile(plots$long,probs=seq(0,1,length.out=5),names=FALSE))
qy <- unique(quantile(plots$lat,probs=seq(0,1,length.out=5),names=FALSE))
stopifnot(length(qx)>=3,length(qy)>=3)
plots$ix <- findInterval(plots$long,qx,all.inside=TRUE)
plots$iy <- findInterval(plots$lat,qy,all.inside=TRUE)
plots$block <- paste(plots$ix,plots$iy,sep="_")
blocks <- as.data.frame(table(plots$block),stringsAsFactors=FALSE)
names(blocks) <- c("block","nplots")
blocks <- blocks[blocks$nplots>0,]
stopifnot(nrow(blocks)>=5)
set.seed(2026)
blocks <- blocks[sample(seq_len(nrow(blocks))),]
blocks <- blocks[order(-blocks$nplots),]
load <- rep(0L,5)
blocks$fold <- integer(nrow(blocks))
for(i in seq_len(nrow(blocks))){
  k <- which.min(load)
  blocks$fold[i] <- k
  load[k] <- load[k]+blocks$nplots[i]
}
plots$fold <- blocks$fold[match(plots$block,blocks$block)]
fold <- plots$fold[match(d$plot,plots$plot)]
stopifnot(!anyNA(fold),length(unique(fold))==5L)
# Fail fast if the spatial assignment violates block/plot integrity.
stopifnot(!anyDuplicated(plots$plot))
stopifnot(all(vapply(split(plots$fold,plots$block),
                     function(z) length(unique(z))==1L,logical(1))))
stopifnot(all(table(factor(fold,levels=1:5))>0))
# Fold diagnostics are exported even before fitting models.
fold_diagnostics <- do.call(rbind,lapply(1:5,function(k){
  idx <- which(fold==k)
  data.frame(fold=k,rows=length(idx),plots=length(unique(d$plot[idx])),
    spatial_blocks=length(unique(plots$block[plots$fold==k])),
    positive_rows=sum(d$pres.abs[idx]==1),
    prevalence=mean(d$pres.abs[idx]),
    longitude_min=min(d$long[idx]),longitude_max=max(d$long[idx]),
    latitude_min=min(d$lat[idx]),latitude_max=max(d$lat[idx]))
}))
write.csv(fold_diagnostics,"results/spatial_2d_block_fold_diagnostics.csv",
          row.names=FALSE)
print(fold_diagnostics)

# Fixed, matched predictors and hyperparameters to the longitude-band study.
for(nm in c("Species","disturb","soilfert")) d[[nm]] <- factor(d[[nm]])
x <- model.matrix(~ . - 1,data=d[,c("rainann","soildepth","tempann","topo",
                                      "Species","disturb","soilfert"),drop=FALSE])
y <- d$pres.abs
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,
               max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
safe <- function(a,b) if(b>0) a/b else NA_real_
out <- vector("list",5); predictions <- vector("list",5)
for(k in 1:5){
  tr <- which(fold!=k); te <- which(fold==k)
  stopifnot(!length(intersect(unique(d$plot[tr]),unique(d$plot[te]))))
  set.seed(100+k)
  fit <- xgb.train(params=params,
    data=xgb.DMatrix(x[tr,,drop=FALSE],label=y[tr]),nrounds=300,verbose=0)
  p <- pmin(pmax(predict(fit,x[te,,drop=FALSE]),1e-15),1-1e-15)
  truth <- y[te]; z <- p>=.5
  tp <- sum(z & truth==1);fp <- sum(z & truth==0)
  tn <- sum(!z & truth==0);fn <- sum(!z & truth==1)
  auc <- if(length(unique(truth))==2) as.numeric(pROC::auc(
    pROC::roc(truth,p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
  ap <- if(sum(truth)>0) {
    o <- order(p,decreasing=TRUE)
    sum((cumsum(truth[o]==1)/seq_along(o))[truth[o]==1])/sum(truth)
  } else NA_real_
  out[[k]] <- data.frame(fold=k,train_rows=length(tr),
    validation_rows=length(te),validation_plots=length(unique(d$plot[te])),
    prevalence=mean(truth),roc_auc=auc,average_precision=ap,
    log_loss=-mean(truth*log(p)+(1-truth)*log1p(-p)),
    precision=safe(tp,tp+fp),recall=safe(tp,tp+fn),
    f1=safe(2*tp,2*tp+fp+fn),tp=tp,fp=fp,tn=tn,fn=fn)
  predictions[[k]] <- data.frame(id=d$id[te],plot=d$plot[te],fold=k,
    observed=truth,predicted_probability=p)
}
metrics <- do.call(rbind,out)
write.csv(metrics,"results/spatial_2d_block_cv_metrics.csv",row.names=FALSE)
write.csv(do.call(rbind,predictions),
          "results/spatial_2d_block_cv_predictions.csv",row.names=FALSE)
summary <- data.frame(folds=5,blocks=nrow(blocks),
 mean_roc_auc=mean(metrics$roc_auc,na.rm=TRUE),
 sd_roc_auc=sd(metrics$roc_auc,na.rm=TRUE),
 mean_average_precision=mean(metrics$average_precision,na.rm=TRUE),
 mean_log_loss=mean(metrics$log_loss),
 mean_recall=mean(metrics$recall,na.rm=TRUE),
 missing_auc_folds=sum(is.na(metrics$roc_auc)))
write.csv(summary,"results/spatial_2d_block_cv_summary.csv",row.names=FALSE)
write.csv(plots[,c("plot","long","lat","ix","iy","block","fold")],
          "results/spatial_2d_block_assignments.csv",row.names=FALSE)
g <- ggplot(plots,aes(long,lat,color=factor(fold)))+
  geom_point(size=1,alpha=.8)+coord_quickmap()+
  labs(title="Two-dimensional geographic-block validation",
       subtitle="Plot-level 4x4 quantile grid; whole grid cells assigned to 5 folds",
       x="Longitude",y="Latitude",color="Held-out fold")+
  theme_minimal(base_size=12)
ggsave("results/figures/16_spatial_2d_block_assignments.png",
       g,width=9,height=7,dpi=160)
g2 <- ggplot(metrics,aes(factor(fold),roc_auc))+
  geom_col(fill="#346c9c")+coord_cartesian(ylim=c(0,1))+
  labs(title="2D spatial-block transfer: ROC AUC by fold",
       x="Held-out spatial fold",y="ROC AUC")+
  theme_minimal(base_size=12)
ggsave("results/figures/17_spatial_2d_block_auc.png",
       g2,width=8,height=5,dpi=160)
print(metrics);print(summary)

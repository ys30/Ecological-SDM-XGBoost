# MaxEnt (maxnet) baseline with spatial-block validation.
# Presence-background model: training background is sampled from observed
# survey plots where focal species was NOT detected. These are survey-derived
# non-detections used as a background approximation, not independently
# sampled landscape background. Output is relative suitability, not calibrated
# probability of occurrence.
suppressPackageStartupMessages({library(maxnet);library(pROC);library(ggplot2)})
d <- read.csv("data/train.csv",stringsAsFactors=FALSE,check.names=FALSE)
a <- read.csv("results/spatial_2d_block_assignments.csv",stringsAsFactors=FALSE)
stopifnot(!anyNA(d),!anyNA(a),!anyDuplicated(d$id),!anyDuplicated(a$plot))
d$fold <- a$fold[match(d$plot,a$plot)]
stopifnot(!anyNA(d$fold),identical(sort(unique(d$fold)),1:5))
stopifnot(all(vapply(split(d$fold,d$plot),function(z) length(unique(z))==1L,logical(1))))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
# Numeric environmental features only; exclude location and taxon ID.
features <- c("rainann","soildepth","tempann","topo")
# Species-specific model is essential for classical MaxEnt.
# Presence/absence is a separate response for each focal taxon.
species <- sort(unique(d$Species))
safe_auc <- function(y,s) if(length(unique(y))==2) as.numeric(pROC::auc(
  pROC::roc(y,s,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
ap <- function(y,s) {
 if(sum(y==1)==0) return(NA_real_)
 o <- order(s,decreasing=TRUE)
 sum((cumsum(y[o]==1)/seq_along(o))[y[o]==1])/sum(y==1)
}
out <- list(); preds <- list(); counter <- 0L
for(sp in species) {
 z <- d[d$Species==sp,,drop=FALSE]
 for(k in 1:5) {
  tr <- z[z$fold!=k,,drop=FALSE]
  te <- z[z$fold==k,,drop=FALSE]
  npos <- sum(tr$pres.abs==1)
  nbg <- sum(tr$pres.abs==0)
  # Rare taxa are explicitly flagged, never silently reported as validated.
  if(npos<10 || nbg<10 || length(unique(te$pres.abs))<2) {
   counter <- counter+1L
   out[[counter]] <- data.frame(species=sp,fold=k,train_presence=npos,
    train_background=nbg,validation_rows=nrow(te),
    validation_presence=sum(te$pres.abs==1),roc_auc=NA_real_,
    average_precision=NA_real_,status="insufficient_positive_records")
   next
  }
  # Survey-derived background includes all non-detections within training folds.
  # maxnet requires p=1 for presence, p=0 for background.
  xtr <- tr[,features,drop=FALSE]
  xte <- te[,features,drop=FALSE]
  # Linear and quadratic features avoid high-complexity hinges for sparse taxa.
  fit <- tryCatch(maxnet::maxnet(p=tr$pres.abs,data=xtr,
    f=maxnet::maxnet.formula(tr$pres.abs,xtr,classes="lq"),
    regmult=2),error=function(e) e)
  if(inherits(fit,"error")) {
   counter <- counter+1L
   out[[counter]] <- data.frame(species=sp,fold=k,train_presence=npos,
    train_background=nbg,validation_rows=nrow(te),
    validation_presence=sum(te$pres.abs==1),roc_auc=NA_real_,
    average_precision=NA_real_,status="fit_error")
   next
  }
  score <- tryCatch(as.numeric(predict(fit,xte,type="cloglog")),
                    error=function(e) rep(NA_real_,nrow(te)))
  if(any(!is.finite(score))) {
   counter <- counter+1L
   out[[counter]] <- data.frame(species=sp,fold=k,train_presence=npos,
    train_background=nbg,validation_rows=nrow(te),
    validation_presence=sum(te$pres.abs==1),roc_auc=NA_real_,
    average_precision=NA_real_,status="prediction_error")
   next
  }
  counter <- counter+1L
  out[[counter]] <- data.frame(species=sp,fold=k,train_presence=npos,
   train_background=nbg,validation_rows=nrow(te),
   validation_presence=sum(te$pres.abs==1),
   roc_auc=safe_auc(te$pres.abs,score),
   average_precision=ap(te$pres.abs,score),status="ok")
  preds[[length(preds)+1L]] <- data.frame(id=te$id,plot=te$plot,
    species=sp,fold=k,observed=te$pres.abs,relative_suitability=score)
 }
}
metrics <- do.call(rbind,out)
write.csv(metrics,"results/maxent_spatial_2d_species_folds.csv",row.names=FALSE)
valid <- metrics[metrics$status=="ok",,drop=FALSE]
summary <- do.call(rbind,lapply(species,function(sp) {
 v <- valid[valid$species==sp,,drop=FALSE]
 data.frame(species=sp,completed_folds=nrow(v),
  mean_roc_auc=if(nrow(v)) mean(v$roc_auc) else NA_real_,
  mean_average_precision=if(nrow(v)) mean(v$average_precision) else NA_real_,
  total_training_presence=min(metrics$train_presence[metrics$species==sp]))
}))
write.csv(summary,"results/maxent_spatial_2d_species_summary.csv",row.names=FALSE)
if(length(preds)) write.csv(do.call(rbind,preds),
 "results/maxent_spatial_2d_predictions.csv",row.names=FALSE)
if(nrow(valid)) {
 g <- ggplot(valid,aes(x=species,y=roc_auc))+
   geom_boxplot(outlier.shape=NA,fill="#87ad99")+
   geom_point(aes(color=factor(fold)),position=position_jitter(width=.12,height=0),size=2)+
   coord_flip(ylim=c(0,1))+theme_minimal(base_size=12)+
   labs(title="MaxEnt: held-out 2D spatial-block discrimination",
        subtitle="Per-species presence vs surveyed non-detection; only estimable folds",
        x="Species",y="ROC AUC",color="Fold")
 ggsave("results/figures/20_maxent_spatial_2d_auc.png",g,width=9,height=6,dpi=160)
}
print(summary)

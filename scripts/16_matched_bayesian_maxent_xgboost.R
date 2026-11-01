suppressPackageStartupMessages(library(pROC))
d <- read.csv("data/train.csv", check.names=FALSE)
b <- read.csv("results/bayes_spatial_cv_predictions.csv")
m <- read.csv("results/maxent_spatial_2d_predictions.csv")
x <- read.csv("results/spatial_2d_block_cv_predictions.csv")
ok <- subset(read.csv("results/maxent_spatial_2d_species_folds.csv"),status=="ok")
stopifnot(!anyDuplicated(d$id),!anyDuplicated(b$id),!anyDuplicated(m$id),!anyDuplicated(x$id))
x$species <- d$Species[match(x$id,d$id)]
names(x)[names(x)=="predicted_probability"] <- "xgboost_score"
stopifnot("xgboost_score" %in% names(x))
keys <- c("id","plot","species","fold","observed")
z <- merge(m,b[,c(keys,"mean_probability","lower_95","upper_95")],by=keys)
z <- merge(z,x[,c(keys,"xgboost_score")],by=keys)
stopifnot(nrow(z)==nrow(m),!anyNA(z$mean_probability),!anyNA(z$xgboost_score))
stopifnot(setequal(unique(paste(z$species,z$fold)),paste(ok$species,ok$fold)))
auc <- function(y,s) as.numeric(pROC::auc(pROC::roc(y,s,levels=c(0,1),direction="<",quiet=TRUE)))
ap <- function(y,s) {
  o <- order(-s,seq_along(s))
  yy <- y[o]==1
  sum((cumsum(yy)/seq_along(yy))[yy])/sum(yy)
}
res <- do.call(rbind,lapply(split(z,paste(z$species,z$fold)),function(v) {
  y <- v$observed
  stopifnot(length(unique(y))==2)
  data.frame(species=v$species[1],fold=v$fold[1],n=nrow(v),positives=sum(y),
    bayesian_auc=auc(y,v$mean_probability),maxent_auc=auc(y,v$relative_suitability),
    xgboost_auc=auc(y,v$xgboost_score),
    bayesian_ap=ap(y,v$mean_probability),maxent_ap=ap(y,v$relative_suitability),
    xgboost_ap=ap(y,v$xgboost_score))
}))
write.csv(res,"results/matched_bayesian_maxent_xgboost_folds.csv",row.names=FALSE)
sp <- do.call(rbind,lapply(split(res,res$species),function(v)
  data.frame(species=v$species[1],folds=nrow(v),n=sum(v$n),positives=sum(v$positives),
    bayesian_auc=mean(v$bayesian_auc),maxent_auc=mean(v$maxent_auc),
    xgboost_auc=mean(v$xgboost_auc),
    bayesian_ap=mean(v$bayesian_ap),maxent_ap=mean(v$maxent_ap),
    xgboost_ap=mean(v$xgboost_ap))))
write.csv(sp,"results/matched_bayesian_maxent_xgboost_species.csv",row.names=FALSE)
print(sp)

suppressPackageStartupMessages({library(pROC);library(ggplot2)})
d <- read.csv("data/train.csv",check.names=FALSE)
x <- read.csv("results/spatial_2d_block_cv_predictions.csv")
m <- read.csv("results/maxent_spatial_2d_predictions.csv")
ok <- subset(read.csv("results/maxent_spatial_2d_species_folds.csv"),status=="ok")
stopifnot(!anyDuplicated(d$id),!anyDuplicated(x$id),!anyDuplicated(m$id))
x$species <- d$Species[match(x$id,d$id)]
names(x)[names(x)=="predicted_probability"] <- "xgboost_score"
z <- merge(m,x[,c("id","plot","fold","species","observed","xgboost_score")],
 by=c("id","plot","fold","species","observed"))
stopifnot(nrow(z)==nrow(m),!anyNA(z$xgboost_score))
stopifnot(setequal(unique(paste(z$species,z$fold)),paste(ok$species,ok$fold)))
ap <- function(y,s) {
 o <- order(-s,seq_along(s)); v <- y[o]==1
 sum((cumsum(v)/seq_along(v))[v])/sum(v)
}
auc <- function(y,s) as.numeric(pROC::auc(pROC::roc(y,s,levels=c(0,1),direction="<",quiet=TRUE)))
res <- do.call(rbind,lapply(split(z,paste(z$species,z$fold)),function(v) {
 y <- v$observed
 stopifnot(length(unique(y))==2)
 data.frame(species=v$species[1],fold=v$fold[1],n=nrow(v),
 maxent_auc=auc(y,v$relative_suitability),xgboost_auc=auc(y,v$xgboost_score),
 maxent_ap=ap(y,v$relative_suitability),xgboost_ap=ap(y,v$xgboost_score))
}))
res$delta_auc <- res$xgboost_auc-res$maxent_auc
res$delta_ap <- res$xgboost_ap-res$maxent_ap
write.csv(res,"results/matched_maxent_xgboost_species_folds.csv",row.names=FALSE)
sp <- do.call(rbind,lapply(split(res,res$species),function(v)
 data.frame(species=v$species[1],folds=nrow(v),
 mean_maxent_auc=mean(v$maxent_auc),mean_xgboost_auc=mean(v$xgboost_auc),
 mean_delta_auc=mean(v$delta_auc),mean_maxent_ap=mean(v$maxent_ap),
 mean_xgboost_ap=mean(v$xgboost_ap),mean_delta_ap=mean(v$delta_ap))))
write.csv(sp,"results/matched_maxent_xgboost_species_summary.csv",row.names=FALSE)
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
g <- ggplot(sp,aes(x=reorder(species,mean_delta_auc),y=mean_delta_auc))+
 geom_col()+coord_flip()+geom_hline(yintercept=0,linetype="dashed")+
 labs(title="Matched spatial folds: XGBoost minus MaxEnt AUC",
 x="Species",y="Mean fold AUC difference")+theme_minimal()
ggsave("results/figures/21_matched_maxent_xgboost_auc.png",g,width=9,height=5)
print(sp)

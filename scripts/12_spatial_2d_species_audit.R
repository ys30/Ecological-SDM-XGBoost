# Species-level audit of existing 2D spatial out-of-fold predictions.
# Descriptive diagnostics: same fitted models, no additional tuning.
suppressPackageStartupMessages(library(ggplot2))
d <- read.csv("data/train.csv",stringsAsFactors=FALSE)
p <- read.csv("results/spatial_2d_block_cv_predictions.csv",stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(d$id),!anyDuplicated(p$id),nrow(p)==nrow(d))
m <- merge(p,d[,c("id","plot","Species","pres.abs")],by="id",all.x=TRUE,
           suffixes=c("_pred","_train"),sort=FALSE)
stopifnot(nrow(m)==nrow(d),!anyNA(m$Species),
          all(m$plot_pred==m$plot_train),
          all(m$observed==m$pres.abs))
safe <- function(x,y) if(y>0) x/y else NA_real_
species <- sort(unique(m$Species))
metrics <- do.call(rbind,lapply(species,function(sp){
 z <- m[m$Species==sp,]
 y <- z$observed; prob <- z$predicted_probability
 pred <- prob>=.5
 tp <- sum(pred & y==1); fp <- sum(pred & y==0)
 tn <- sum(!pred & y==0); fn <- sum(!pred & y==1)
 # Rank-based AUC with ties averaged; only defined for both classes.
 auc <- if(sum(y==1)>0 && sum(y==0)>0) {
  (sum(rank(prob,ties.method="average")[y==1])-
   sum(y==1)*(sum(y==1)+1)/2)/(sum(y==1)*sum(y==0))
 } else NA_real_
 ap <- if(sum(y==1)>0){
  o <- order(-prob,seq_along(prob))
  sum((cumsum(y[o]==1)/seq_along(o))[y[o]==1])/sum(y)
 } else NA_real_
 data.frame(Species=sp,n=nrow(z),positive=sum(y),prevalence=mean(y),
   roc_auc=auc,average_precision=ap,
   brier=mean((prob-y)^2),mean_probability=mean(prob),
   precision=safe(tp,tp+fp),recall=safe(tp,tp+fn),
   f1=safe(2*tp,2*tp+fp+fn),tp=tp,fp=fp,tn=tn,fn=fn)
}))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
write.csv(metrics,"results/spatial_2d_species_metrics.csv",row.names=FALSE)
g <- ggplot(metrics,aes(reorder(Species,average_precision),average_precision))+
 geom_col(fill="#346c9c")+coord_flip()+
 labs(title="Species-level average precision under 2D spatial CV",
      subtitle="Out-of-fold predictions; compare with each species prevalence",
      x="Species",y="Average precision")+
 geom_point(aes(y=prevalence),shape=4,color="#ba4b33",size=2.8)+
 theme_minimal(base_size=12)
ggsave("results/figures/18_spatial_2d_species_ap.png",g,width=9,height=6,dpi=160)
g2 <- ggplot(metrics,aes(reorder(Species,recall),recall))+
 geom_col(fill="#347d72")+coord_flip(ylim=c(0,1))+
 labs(title="Species-level recall at 0.50 threshold",
      subtitle="Out-of-fold 2D spatial-block predictions; not threshold optimized",
      x="Species",y="Recall")+
 theme_minimal(base_size=12)
ggsave("results/figures/19_spatial_2d_species_recall.png",g2,width=9,height=6,dpi=160)
print(metrics)

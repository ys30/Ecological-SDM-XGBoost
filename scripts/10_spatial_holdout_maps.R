# Auditable spatial visualization of observed and predicted occurrences.
# Uses actual plot coordinates from training data and independently saved
# plot-held-out predictions. No basemap or administrative boundaries inferred.
suppressPackageStartupMessages(library(ggplot2))
d <- read.csv("data/train.csv",stringsAsFactors=FALSE)
p <- read.csv("results/plot_holdout_predictions.csv",stringsAsFactors=FALSE)
stopifnot(all(c("id","plot","long","lat","Species","pres.abs") %in% names(d)),
          all(c("id","plot","observed","predicted_probability") %in% names(p)),
          !anyDuplicated(d$id),!anyDuplicated(p$id))
m <- merge(p,d[,c("id","plot","long","lat","Species","pres.abs")],
           by="id",all.x=TRUE,sort=FALSE,suffixes=c("_pred","_train"))
stopifnot(nrow(m)==nrow(p),!anyNA(m$long),!anyNA(m$lat),
          all(m$observed==m$pres.abs),all(m$plot_pred==m$plot_train))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
# Species-specific coordinates are repeated at each plot, hence facets.
g <- ggplot(m,aes(long,lat,color=predicted_probability))+
  geom_point(size=.85,alpha=.8)+
  facet_wrap(~Species,ncol=4)+
  scale_color_viridis_c(option="plasma",limits=c(0,1))+
  coord_quickmap()+labs(title="Held-out plot predictions by tree species",
  subtitle="Observed plot coordinates; color is predicted presence probability",
  x="Longitude",y="Latitude",color="P(presence)")+
  theme_minimal(base_size=11)
ggsave("results/figures/14_spatial_predicted_presence.png",g,width=13,height=8,dpi=160)
m$classification <- factor(ifelse(m$observed==1 & m$predicted_probability>=.5,"TP",
 ifelse(m$observed==1,"FN",ifelse(m$predicted_probability>=.5,"FP","TN"))),
 levels=c("TN","TP","FP","FN"))
g2 <- ggplot(m,aes(long,lat,color=classification))+
  geom_point(size=.9,alpha=.8)+facet_wrap(~Species,ncol=4)+
  scale_color_manual(values=c(TN="#c0c0c0",TP="#228b55",FP="#d49b21",FN="#c43b3b"))+
  coord_quickmap()+labs(title="Spatial distribution of holdout errors (threshold 0.5)",
  subtitle="FN: missed occurrence; FP: false alarm; TP/TN: correct classification",
  x="Longitude",y="Latitude",color="Classification")+
  theme_minimal(base_size=11)
ggsave("results/figures/15_spatial_holdout_errors.png",g2,width=13,height=8,dpi=160)
write.csv(m[,c("id","plot_pred","Species","long","lat","observed",
               "predicted_probability","classification")],
          "results/spatial_holdout_audit.csv",row.names=FALSE)
print(table(m$classification))

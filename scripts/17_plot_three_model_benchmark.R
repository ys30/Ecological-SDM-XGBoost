suppressPackageStartupMessages(library(ggplot2))
sp <- read.csv("results/matched_bayesian_maxent_xgboost_species.csv")
stopifnot(nrow(sp)>0,all(c("species","bayesian_auc","maxent_auc","xgboost_auc","bayesian_ap","maxent_ap","xgboost_ap") %in% names(sp)))
long <- do.call(rbind,lapply(c("bayesian","maxent","xgboost"),function(model) {
  data.frame(species=sp$species,model=model,AUC=sp[[paste0(model,"_auc")]],AP=sp[[paste0(model,"_ap")]])
}))
long$model <- factor(long$model,levels=c("bayesian","maxent","xgboost"),labels=c("Bayesian","MaxEnt","XGBoost"))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
for(metric in c("AUC","AP")) {
  p <- ggplot(long,aes(x=species,y=.data[[metric]],fill=model))+
    geom_col(position=position_dodge(width=.8),width=.72)+
    coord_flip()+scale_y_continuous(limits=c(0,1))+
    labs(title=paste("Matched spatial CV:",metric,"by species"),subtitle="Same eligible species-fold test records for all three models",x=NULL,y=metric,fill="Model")+
    theme_minimal(base_size=12)+theme(legend.position="bottom")
  ggsave(paste0("results/figures/22_three_model_",tolower(metric),".png"),p,width=10,height=6,dpi=180)
}
write.csv(long,"results/matched_three_model_plot_data.csv",row.names=FALSE)

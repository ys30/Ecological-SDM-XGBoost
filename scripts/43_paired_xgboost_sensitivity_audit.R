# Paired species-fold and species-level sensitivity audit (no manuscript files).
d <- read.csv("results/xgboost_matched_23_cell_sensitivity.csv")
stopifnot(nrow(d)==69L,all(c("mode","species","fold","auc","ap","maxent_auc","bayesian_auc","archived_xgboost_auc") %in% names(d)))
modes <- c("fold_local_full","common_four_plus_species","common_four_environment")
stopifnot(setequal(unique(d$mode),modes))
d$delta_auc_vs_maxent <- d$auc-d$maxent_auc
d$delta_auc_vs_bayesian <- d$auc-d$bayesian_auc
d$delta_auc_vs_archived_xgboost <- d$auc-d$archived_xgboost_auc
d$delta_ap_vs_maxent <- d$ap-d$maxent_ap
d$delta_ap_vs_bayesian <- d$ap-d$bayesian_ap
d$delta_ap_vs_archived_xgboost <- d$ap-d$archived_xgboost_ap
write.csv(d,"results/xgboost_matched_23_cell_paired_deltas.csv",row.names=FALSE)
num <- c("auc","ap",grep("^delta_",names(d),value=TRUE))
sp <- aggregate(d[num],by=list(mode=d$mode,species=d$species),FUN=mean)
# Check the number of folds without relying on table ordering.
sp$folds <- vapply(seq_len(nrow(sp)),function(i)sum(d$mode==sp$mode[i] & d$species==sp$species[i]),integer(1))
write.csv(sp,"results/xgboost_matched_species_paired_deltas.csv",row.names=FALSE)
summary <- do.call(rbind,lapply(split(d,d$mode),function(z) {
  data.frame(mode=z$mode[1],cells=nrow(z),species=length(unique(z$species)),
    mean_delta_auc_vs_maxent=mean(z$delta_auc_vs_maxent),
    mean_delta_auc_vs_bayesian=mean(z$delta_auc_vs_bayesian),
    mean_delta_auc_vs_archived_xgboost=mean(z$delta_auc_vs_archived_xgboost),
    mean_delta_ap_vs_maxent=mean(z$delta_ap_vs_maxent),
    mean_delta_ap_vs_bayesian=mean(z$delta_ap_vs_bayesian),
    mean_delta_ap_vs_archived_xgboost=mean(z$delta_ap_vs_archived_xgboost),
    auc_cells_above_maxent=sum(z$delta_auc_vs_maxent>0),
    auc_cells_above_bayesian=sum(z$delta_auc_vs_bayesian>0))
}))
write.csv(summary,"results/xgboost_matched_23_cell_paired_summary.csv",row.names=FALSE)
print(summary,row.names=FALSE)
print(sp[,c("mode","species","folds","auc","ap","delta_auc_vs_maxent","delta_auc_vs_bayesian")],row.names=FALSE)
# Descriptive fold-block bootstrap, preserving species cells within each spatial fold.
# Only five spatial folds: intervals are sensitivity descriptors, not confirmatory CIs.
set.seed(4301)
B <- 2000L
boot <- do.call(rbind,lapply(split(d,d$mode),function(z) {
  vals <- replicate(B,{
    folds <- sample(sort(unique(z$fold)),size=5L,replace=TRUE)
    zz <- do.call(rbind,lapply(folds,function(f)z[z$fold==f,,drop=FALSE]))
    colMeans(zz[,c("delta_auc_vs_maxent","delta_auc_vs_bayesian","delta_ap_vs_maxent","delta_ap_vs_bayesian")])
  })
  data.frame(mode=z$mode[1],metric=rownames(vals),
    low=apply(vals,1,quantile,probs=.025),high=apply(vals,1,quantile,probs=.975),
    method="descriptive_fold_block_bootstrap_2000")
}))
write.csv(boot,"results/xgboost_matched_fold_bootstrap_descriptive.csv",row.names=FALSE)
print(boot,row.names=FALSE)

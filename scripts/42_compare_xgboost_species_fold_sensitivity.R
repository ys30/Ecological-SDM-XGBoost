# Compare sensitivity XGBoost models on the exact 23 species-fold cells
# used in the archived Bayesian/MaxEnt/XGBoost matched evaluation.
s <- read.csv("results/xgboost_spatial_species_fold_sensitivity.csv")
m <- read.csv("results/matched_bayesian_maxent_xgboost_folds.csv")
stopifnot(!anyDuplicated(m[c("species","fold")]),nrow(m)==23L,
          all(c("mode","species","fold","n","positive","auc","ap","status") %in% names(s)))
modes <- c("fold_local_full","common_four_plus_species","common_four_environment")
stopifnot(setequal(unique(s$mode),modes))
out <- list()
for (mode in modes) {
  z <- merge(m,subset(s,mode==mode),by=c("species","fold"),all.x=TRUE,sort=FALSE)
  stopifnot(nrow(z)==nrow(m),!anyNA(z$mode),all(z$status=="ok"),
            all(z$n==z$n.x | !("n.x" %in% names(z))))
  # Exact held-out count and positive count audit
  stopifnot(all(z$n.x==z$n.y),all(z$positives==z$positive))
  out[[mode]] <- data.frame(mode=mode,species=z$species,fold=z$fold,
    n=z$n.x,positives=z$positives,auc=z$auc,ap=z$ap,
    archived_xgboost_auc=z$xgboost_auc,archived_xgboost_ap=z$xgboost_ap,
    maxent_auc=z$maxent_auc,maxent_ap=z$maxent_ap,
    bayesian_auc=z$bayesian_auc,bayesian_ap=z$bayesian_ap)
}
v <- do.call(rbind,out)
stopifnot(nrow(v)==69L)
write.csv(v,"results/xgboost_matched_23_cell_sensitivity.csv",row.names=FALSE)
summary <- aggregate(cbind(auc,ap,archived_xgboost_auc,archived_xgboost_ap,
                           maxent_auc,maxent_ap,bayesian_auc,bayesian_ap)~mode,
                     data=v,FUN=mean)
write.csv(summary,"results/xgboost_matched_23_cell_summary.csv",row.names=FALSE)
print(summary,row.names=FALSE)

# EMS v6.14: uncertainty for matched three-way ablation
# Resample taxa (not individual plot-taxon rows); small-cluster intervals are descriptive.
x <- read.csv("results/v614_common_input/three_way_xgboost_ablation.csv")
stopifnot(all(c("species","fold","delta_pooled_minus_specific_auc",
 "delta_blind_minus_pooled_auc") %in% names(x)))
set.seed(614)
metrics <- grep("^delta_",names(x),value=TRUE)
rows <- list()
for(metric in metrics) {
 z <- x[is.finite(x[[metric]]),c("species","fold",metric),drop=FALSE]
 taxa <- sort(unique(z$species))
 if(!nrow(z))next
 # First average within each taxon, then equally weight taxa.
 tm <- vapply(taxa,function(sp)mean(z[z$species==sp,metric]),numeric(1))
 observed <- mean(tm)
 B <- 10000L
 boot <- replicate(B,mean(sample(tm,length(tm),replace=TRUE)))
 # Exact two-sided sign-flip reference over taxon-level means.
 signs <- as.matrix(expand.grid(rep(list(c(-1,1)),length(tm))))
 null <- as.vector(signs %*% tm)/length(tm)
 p <- mean(abs(null)>=abs(observed)-1e-12)
 rows[[length(rows)+1L]] <- data.frame(
  comparison=metric,n_taxa=length(tm),n_cells=nrow(z),
  equal_taxon_mean=observed,bootstrap_lower=unname(quantile(boot,.025)),
  bootstrap_upper=unname(quantile(boot,.975)),signflip_tail_fraction=p,
  interpretation="exploratory; taxon clusters are few")
}
out <- do.call(rbind,rows)
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(out,"results/v614_common_input/three_way_taxon_cluster_uncertainty.csv",row.names=FALSE)
print(out)

# EMS v6.14: pooled XGBoost with the same four environmental predictors + species ID
# Paired against taxon-specific XGBoost on identical plot-held-out folds.
suppressPackageStartupMessages({library(xgboost);library(pROC)})
d <- read.csv("data/train.csv",check.names=FALSE)
f <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- f$fold[match(d$plot,f$plot)]
env <- c("rainann","soildepth","tempann","topo")
stopifnot(!anyNA(d$fold),!anyNA(d[,env]),!anyDuplicated(d$id))
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,
 max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
rows <- list()
for(k in sort(unique(d$fold))) {
 tr <- d[d$fold!=k,,drop=FALSE];te <- d[d$fold==k,,drop=FALSE]
 stopifnot(!length(intersect(unique(tr$plot),unique(te$plot))))
 stopifnot(length(unique(tr$pres.abs))==2)
 species_levels <- sort(unique(as.character(tr$Species)))
 stopifnot(all(te$Species %in% species_levels))
 make_matrix <- function(z) {
   mat <- as.matrix(z[,env,drop=FALSE])
   for(sp in species_levels) mat <- cbind(mat,as.integer(as.character(z$Species)==sp))
   colnames(mat) <- c(env,paste0("Species_",species_levels))
   storage.mode(mat)<-"double"
   mat
 }
 set.seed(1000+k)
 fit <- xgb.train(params=params,data=xgb.DMatrix(make_matrix(tr),label=tr$pres.abs),
                  nrounds=300,verbose=0)
 pp <- pmin(pmax(as.numeric(predict(fit,xgb.DMatrix(make_matrix(te)))),1e-15),1-1e-15)
 rows[[length(rows)+1L]] <- data.frame(id=te$id,plot=te$plot,Species=te$Species,
    fold=k,y=te$pres.abs,p=pp)
}
out <- do.call(rbind,rows)
stopifnot(nrow(out)==nrow(d),setequal(out$id,d$id),!anyDuplicated(out$id),
          all(is.finite(out$p)),all(out$p>0 & out$p<1))
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(out,"results/v614_common_input/pooled_four_plus_species_xgboost_oof.csv",row.names=FALSE)
cat("Pooled common-input OOF predictions:",nrow(out),"\n")

# Numerical comparison of historical R fixed XGBoost and Python fixed OOF.
# Uses historical R grid4 output, joined by plot/taxon/fold to frozen Python OOF.
suppressPackageStartupMessages(library(xgboost))
d <- read.csv("data/train.csv", check.names=FALSE)
f <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- f$fold[match(d$plot,f$plot)]
stopifnot(!anyNA(d$fold), all(c("rainann","soildepth","tempann","topo") %in% names(d)))
env <- c("rainann","soildepth","tempann","topo")
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,max_depth=4,
               subsample=.8,colsample_bytree=.8,min_child_weight=1,nthread=2,tree_method="hist")
dir.create("results/v622",recursive=TRUE,showWarnings=FALSE)
cat("R_VERSION:",R.version.string,"\nXGBOOST_R_VERSION:",as.character(packageVersion("xgboost")),"\n")
rows <- list()
for (k in sort(unique(d$fold))) {
  tr <- d[d$fold!=k,,drop=FALSE]; te <- d[d$fold==k,,drop=FALSE]
  stopifnot(!length(intersect(unique(tr$plot),unique(te$plot))))
  levels <- sort(unique(as.character(tr$Species)))
  make_mat <- function(z, pooled=FALSE) {
    m <- as.matrix(z[,env,drop=FALSE])
    if (pooled) {
      for (sp in levels) m <- cbind(m,as.integer(as.character(z$Species)==sp))
      colnames(m) <- c(env,paste0("Species_",levels))
    }
    storage.mode(m)<-"double"
    m
  }
  for (variant in c("species","pooled")) {
    for (sp in if(variant=="species") levels else "ALL") {
      train <- if(variant=="species") tr[tr$Species==sp,,drop=FALSE] else tr
      test <- if(variant=="species") te[te$Species==sp,,drop=FALSE] else te
      if (variant=="species" && (sum(train$pres.abs)<5 || length(unique(train$pres.abs))<2)) next
      if(length(unique(train$pres.abs))<2) stop("pooled lacks both classes")
      set.seed(1000+k)
      fit <- xgb.train(params=params,data=xgb.DMatrix(make_mat(train,variant=="pooled"),label=train$pres.abs),
                       nrounds=300,verbose=0)
      pred <- as.numeric(predict(fit,xgb.DMatrix(make_mat(test,variant=="pooled"))))
      rows[[length(rows)+1L]] <- data.frame(plot=test$plot,Species=test$Species,
        outer_fold=k,variant=variant,p_r=pred)
    }
  }
}
out <- do.call(rbind,rows)
stopifnot(!anyDuplicated(out[,c("plot","Species","outer_fold","variant")]))
write.csv(out,"results/v622/c1_r_fixed_hist_oof_grid4.csv",row.names=FALSE)
cat("R_PARITY_ROWS",nrow(out),"\n")

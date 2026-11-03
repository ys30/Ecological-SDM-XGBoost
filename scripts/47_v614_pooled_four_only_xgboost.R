# EMS v6.14 pooled XGBoost without species identity: negative-control ablation.
# A shared prediction function for all taxa is NOT taxon-specific SDM.
suppressPackageStartupMessages(library(xgboost))
d <- read.csv("data/train.csv",check.names=FALSE)
f <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- f$fold[match(d$plot,f$plot)]
env <- c("rainann","soildepth","tempann","topo")
stopifnot(!anyNA(d$fold),!anyNA(d[,env]),!anyDuplicated(d$id))
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,
 max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
rows <- list()
for(k in sort(unique(d$fold))) {
 tr<-d[d$fold!=k,,drop=FALSE];te<-d[d$fold==k,,drop=FALSE]
 stopifnot(!length(intersect(unique(tr$plot),unique(te$plot))),
           length(unique(tr$pres.abs))==2)
 set.seed(1000+k)
 fit<-xgb.train(params=params,
   data=xgb.DMatrix(as.matrix(tr[,env,drop=FALSE]),label=tr$pres.abs),
   nrounds=300,verbose=0)
 pr<-pmin(pmax(as.numeric(predict(fit,
   xgb.DMatrix(as.matrix(te[,env,drop=FALSE])))),1e-15),1-1e-15)
 rows[[length(rows)+1L]]<-data.frame(id=te$id,plot=te$plot,
   Species=te$Species,fold=k,y=te$pres.abs,p=pr)
}
out<-do.call(rbind,rows)
stopifnot(nrow(out)==nrow(d),setequal(out$id,d$id),
 !anyDuplicated(out$id),all(is.finite(out$p)))
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(out,"results/v614_common_input/pooled_four_only_xgboost_oof.csv",row.names=FALSE)
cat("Species-blind pooled OOF rows:",nrow(out),"\n")

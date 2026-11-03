# EMS v6.14 common-four-predictor taxon-specific XGBoost
# Uses fixed existing spatial folds; does not overwrite prior results.
suppressPackageStartupMessages({library(xgboost);library(pROC)})
d <- read.csv("data/train.csv", check.names=FALSE)
f <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- f$fold[match(d$plot,f$plot)]
stopifnot(!anyNA(d$fold), !anyDuplicated(d$id), all(d$pres.abs %in% c(0,1)))
env <- c("rainann","soildepth","tempann","topo")
stopifnot(all(env %in% names(d)), !anyNA(d[,env]))
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
rows <- list()
for (sp in sort(unique(d$Species))) for (k in sort(unique(d$fold))) {
 tr <- d[d$Species==sp & d$fold!=k,,drop=FALSE]
 te <- d[d$Species==sp & d$fold==k,,drop=FALSE]
 stopifnot(length(intersect(tr$plot,te$plot))==0)
 if (length(unique(tr$pres.abs))<2) {
   rows[[length(rows)+1L]] <- data.frame(id=te$id,plot=te$plot,Species=sp,fold=k,y=te$pres.abs,p=NA_real_,status="single_class_training")
   next
 }
 set.seed(1000+k)
 fit <- xgb.train(params=params,data=xgb.DMatrix(as.matrix(tr[,env]),label=tr$pres.abs),nrounds=300,verbose=0)
 pp <- pmin(pmax(as.numeric(predict(fit,xgb.DMatrix(as.matrix(te[,env])))),1e-15),1-1e-15)
 rows[[length(rows)+1L]] <- data.frame(id=te$id,plot=te$plot,Species=sp,fold=k,y=te$pres.abs,p=pp,status="ok")
}
out <- do.call(rbind,rows)
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(out,"results/v614_common_input/taxon_specific_xgboost_oof.csv",row.names=FALSE)
metric <- function(z) {
 z <- z[is.finite(z$p),,drop=FALSE]
 if(!nrow(z)) return(NULL)
 auc <- if(length(unique(z$y))==2) as.numeric(pROC::auc(pROC::roc(z$y,z$p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
 ap <- if(sum(z$y)>0) {a<-z[order(-z$p),,drop=FALSE];sum((cumsum(a$y)/seq_len(nrow(a)))[a$y==1])/sum(a$y)} else NA_real_
 data.frame(Species=z$Species[1],fold=z$fold[1],n=nrow(z),positives=sum(z$y),auc=auc,ap=ap,brier=mean((z$y-z$p)^2),logloss=-mean(z$y*log(z$p)+(1-z$y)*log1p(-z$p)))
}
groups <- split(out,list(out$Species,out$fold),drop=TRUE)
metrics <- do.call(rbind,lapply(groups,metric))
write.csv(metrics,"results/v614_common_input/taxon_specific_xgboost_metrics.csv",row.names=FALSE)
print(metrics)

# Sensitivity analysis: fold-local encoding and four-environment-predictor XGBoost.
# Does not overwrite archived baseline or manuscript results.
suppressPackageStartupMessages({library(xgboost); library(pROC)})
d <- read.csv("data/train.csv",stringsAsFactors=FALSE,check.names=FALSE)
a <- read.csv("results/spatial_2d_block_assignments.csv",stringsAsFactors=FALSE)
d$fold <- a$fold[match(d$plot,a$plot)]
stopifnot(!anyNA(d$fold),!anyNA(d),!anyDuplicated(d$id),all(d$pres.abs %in% c(0,1)))
env <- c("rainann","soildepth","tempann","topo")
cats <- c("Species","disturb","soilfert")
# Original pooled XGBoost uses 4 environmental + 3 categorical fields;
# dummy expansion produces more than seven matrix columns.
auc <- function(y,p) as.numeric(pROC::auc(pROC::roc(y,p,levels=c(0,1),direction="<",quiet=TRUE)))
ap <- function(y,p) {o<-order(p,decreasing=TRUE); z<-y[o]==1; sum((cumsum(z)/seq_along(z))[z])/sum(z)}
params <- list(objective="binary:logistic",eval_metric="logloss",eta=.03,
               max_depth=4,subsample=.8,colsample_bytree=.8,nthread=2)
out <- list(); species_out <- list()
for(k in 1:5) {
 tr <- d[d$fold!=k,,drop=FALSE]; te <- d[d$fold==k,,drop=FALSE]
 stopifnot(length(intersect(unique(tr$plot),unique(te$plot)))==0)
 # Learn categorical schema exclusively from training rows. Unknown test
 # levels become an all-zero indicator group (explicit reference fallback).
 make_x <- function(features, categorical=character()) {
   train <- tr[,features,drop=FALSE]; test <- te[,features,drop=FALSE]
   if(length(categorical)) for(nm in categorical) {
     lv <- sort(unique(as.character(train[[nm]])))
     train[[nm]] <- factor(train[[nm]],levels=lv)
     test[[nm]] <- factor(as.character(test[[nm]]),levels=lv)
   }
   if(!length(categorical)) return(list(train=as.matrix(train),test=as.matrix(test),
                                    unseen=0L))
   # Explicit deterministic one-hot columns with train-only category vocabulary.
   tx <- as.matrix(train[,setdiff(features,categorical),drop=FALSE])
   vx <- as.matrix(test[,setdiff(features,categorical),drop=FALSE])
   unseen <- 0L
   for(nm in categorical) {
     lv <- levels(train[[nm]])
     unseen <- unseen + sum(is.na(test[[nm]]))
     for(label in lv) {
       col <- paste0(nm,"_",make.names(label))
       tx <- cbind(tx,as.integer(as.character(train[[nm]])==label))
       vx <- cbind(vx,as.integer(!is.na(test[[nm]]) & as.character(test[[nm]])==label))
       colnames(tx)[ncol(tx)] <- col
       colnames(vx)[ncol(vx)] <- col
     }
   }
   stopifnot(identical(colnames(tx),colnames(vx)),!anyNA(tx),!anyNA(vx))
   list(train=tx,test=vx,unseen=unseen)
 }
 for(mode in c("fold_local_full","common_four_environment","common_four_plus_species")) {
   features <- if(mode=="fold_local_full") c(env,cats) else if(mode=="common_four_plus_species") c(env,"Species") else env
   enc <- make_x(features,if(mode=="fold_local_full") cats else if(mode=="common_four_plus_species") "Species" else character())
   set.seed(100+k)
   fit <- xgb.train(params=params,data=xgb.DMatrix(enc$train,label=tr$pres.abs),
                    nrounds=300,verbose=0)
   p <- pmin(pmax(as.numeric(predict(fit,enc$test)),1e-15),1-1e-15)
   for(sp in sort(unique(as.character(te$Species)))) {
     ix <- as.character(te$Species)==sp
     yy <- te$pres.abs[ix]; pp <- p[ix]
     species_out[[length(species_out)+1L]] <- data.frame(mode=mode,fold=k,species=sp,
       n=sum(ix),positive=sum(yy),
       auc=if(length(unique(yy))==2) auc(yy,pp) else NA_real_,
       ap=if(sum(yy)>0) ap(yy,pp) else NA_real_,
       status=if(length(unique(yy))==2) "ok" else "insufficient_test_classes")
   }
   out[[length(out)+1L]] <- data.frame(mode=mode,fold=k,
     n=nrow(te),positive=sum(te$pres.abs),unseen_category_values=enc$unseen,
     auc=auc(te$pres.abs,p),ap=ap(te$pres.abs,p),
     logloss=-mean(te$pres.abs*log(p)+(1-te$pres.abs)*log1p(-p)))
 }
}
res <- do.call(rbind,out)
write.csv(res,"results/xgboost_spatial_encoding_sensitivity.csv",row.names=FALSE)
print(res)

write.csv(do.call(rbind,species_out),"results/xgboost_spatial_species_fold_sensitivity.csv",row.names=FALSE)
print(aggregate(cbind(auc,ap)~mode,data=subset(do.call(rbind,species_out),status=="ok"),FUN=mean))

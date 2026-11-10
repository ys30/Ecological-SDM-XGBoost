#!/usr/bin/env Rscript
# EMS v6.22 C1 runtime-only pilot. Does NOT evaluate or write outer test predictions.
suppressPackageStartupMessages({library(xgboost); library(stats)})
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<1L) stop("Usage: Rscript scripts/91_v622_c1_runtime_pilot.R <approved-fold-csv> [output-dir]")
fold_file <- args[1]; output_dir <- if(length(args)>1L) args[2] else "results/v622/pilot"
d <- read.csv("data/train.csv", check.names=FALSE, stringsAsFactors=FALSE)
f <- read.csv(fold_file, stringsAsFactors=FALSE)
stopifnot(all(c("plot","fold") %in% names(f)), !anyDuplicated(f$plot),
          all(c("plot","Species","pres.abs","rainann","soildepth","tempann","topo","easting","northing") %in% names(d)),
          !anyDuplicated(d$id), all(d$pres.abs %in% 0:1))
stopifnot(setequal(unique(d$plot),f$plot), !anyNA(d[,c("rainann","soildepth","tempann","topo","easting","northing")]))
d$outer_fold <- f$fold[match(d$plot,f$plot)]
outer <- sort(unique(d$outer_fold))[1]
train <- d[d$outer_fold != outer,,drop=FALSE]
test_plots <- unique(d$plot[d$outer_fold == outer])
stopifnot(!length(intersect(unique(train$plot),test_plots)))
plots <- train[!duplicated(train$plot),c("plot","easting","northing")]
plots <- plots[order(plots$plot),,drop=FALSE]
xy <- scale(as.matrix(plots[,c("easting","northing")]))
stopifnot(all(is.finite(xy)),nrow(plots)>=12L)
set.seed(6220L+as.integer(outer))
km <- kmeans(xy,centers=12L,nstart=30L,iter.max=100L)
# Allocate whole clusters to 4 inner folds, balancing plot counts.
counts <- sort(table(km$cluster),decreasing=TRUE)
loads <- integer(4); cluster_fold <- integer(12)
for(cl in names(counts)) {k <- which.min(loads);cluster_fold[as.integer(cl)] <- k;loads[k] <- loads[k]+as.integer(counts[cl])}
plots$inner_fold <- cluster_fold[km$cluster]
train$inner_fold <- plots$inner_fold[match(train$plot,plots$plot)]
stopifnot(!anyNA(train$inner_fold),length(unique(train$inner_fold))==4L)
set.seed(622L)
candidates <- data.frame(
  candidate=seq_len(30L),max_depth=sample(2:7,30,TRUE),
  eta=sample(c(.01,.03,.05,.1),30,TRUE),
  min_child_weight=sample(c(1,3,5,10),30,TRUE),
  subsample=sample(c(.65,.8,1),30,TRUE),
  colsample_bytree=sample(c(.65,.8,1),30,TRUE),
  nrounds=sample(c(100L,200L,300L,500L),30,TRUE))
env <- c("rainann","soildepth","tempann","topo")
taxa <- sort(unique(train$Species))
clip <- function(p) pmin(pmax(p,1e-15),1-1e-15)
ll <- function(y,p) -mean(y*log(clip(p))+(1-y)*log1p(-clip(p)))
# Pooled and per-taxon variants; both use exactly the four environmental predictors.
make_x <- function(z,pooled) {
  x <- as.matrix(z[,env,drop=FALSE]);storage.mode(x)<-"double"
  if(pooled) {s <- outer(z$Species,taxa,`==`)*1.0;colnames(s)<-paste0("taxon_",taxa);x<-cbind(x,s)}
  x
}
# Runtime-only: never score outer held-out records.
start <- Sys.time(); log_rows <- list()
for(pooled in c(FALSE,TRUE)) for(i in seq_len(nrow(candidates))) {
  c <- candidates[i,]; scores <- list(); t0 <- Sys.time()
  for(k in sort(unique(train$inner_fold))) {
    tr <- train[train$inner_fold!=k,,drop=FALSE]
    va <- train[train$inner_fold==k,,drop=FALSE]
    stopifnot(!length(intersect(unique(tr$plot),unique(va$plot))))
    taxon_scores <- numeric()
    for(sp in taxa) {
      a <- if(pooled) tr else tr[tr$Species==sp,,drop=FALSE]
      b <- va[va$Species==sp,,drop=FALSE]
      if(!nrow(b)) next
      prevalence <- mean(a$pres.abs)
      if(!is.finite(prevalence)) next
      if(sum(a$pres.abs)<5L || length(unique(a$pres.abs))<2L) {
        pred <- rep(prevalence,nrow(b))
      } else {
        pars <- list(objective="binary:logistic",eval_metric="logloss",max_depth=as.integer(c$max_depth),
          eta=c$eta,min_child_weight=c$min_child_weight,subsample=c$subsample,
          colsample_bytree=c$colsample_bytree,nthread=2L)
        fit <- xgb.train(params=pars,data=xgb.DMatrix(make_x(a,pooled),label=a$pres.abs),
                         nrounds=as.integer(c$nrounds),verbose=0)
        pred <- as.numeric(predict(fit,xgb.DMatrix(make_x(b,pooled))))
      }
      null_ll <- ll(b$pres.abs,rep(prevalence,nrow(b)))
      if(null_ll>0) taxon_scores <- c(taxon_scores,1-ll(b$pres.abs,pred)/null_ll)
    }
    if(length(taxon_scores)) scores[[length(scores)+1L]] <- mean(taxon_scores)
  }
  log_rows[[length(log_rows)+1L]] <- data.frame(variant=if(pooled)"pooled" else "per_taxon",
      candidate=i,inner_mean_relative_logloss=if(length(scores))mean(unlist(scores)) else NA_real_,
      elapsed_seconds=as.numeric(difftime(Sys.time(),t0,units="secs")))
}
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
# No outer predictions or metrics are created. Inner candidate scores are tuning diagnostics only.
write.csv(candidates,file.path(output_dir,"candidate_grid.csv"),row.names=FALSE)
write.csv(do.call(rbind,log_rows),file.path(output_dir,"inner_candidate_runtime.csv"),row.names=FALSE)
write.csv(data.frame(design_fold_file=fold_file,outer_fold=outer,
  elapsed_seconds=as.numeric(difftime(Sys.time(),start,units="secs")),
  outer_test_rows_not_scored=sum(d$outer_fold==outer)),
  file.path(output_dir,"runtime_summary.csv"),row.names=FALSE)
cat("Runtime-only pilot finished; outer test rows were never scored.\n")

suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_r_native"
records <- list(); winners <- list()
for (model in c("glmnet_lin","glmnet_quad_rt")) {
  # Candidate paths use outer-training observations only. A single numeric
  # lambda grid per alpha is passed unchanged to all four inner folds.
  parts <- lapply(0:3,function(f) read.csv(sprintf("%s/%s_fold%d_valid.csv",base,model,f),check.names=FALSE))
  outer_train <- do.call(rbind,parts)
  if (anyDuplicated(outer_train)) warning("duplicate rows: confirm spatial partition integrity")
  xx <- as.matrix(outer_train[,setdiff(names(outer_train),"y"),drop=FALSE])
  yy <- outer_train$y
  for (alpha in c(0,0.5,1)) {
    # glmnet's automatic path may terminate early: derive the 30-value
    # candidate grid from the outer-training-only full path's endpoints.
    auto <- glmnet(xx,yy,family="binomial",alpha=alpha,nlambda=30,
                   standardize=FALSE,intercept=TRUE,thresh=1e-8,maxit=100000)
    if(length(auto$lambda)<2) stop("degenerate candidate path")
    grid <- exp(seq(log(max(auto$lambda)),log(min(auto$lambda)),length.out=30))
    if(length(unique(grid))!=30) stop("nonunique lambda grid")
    for(fold in 0:3) {
      tr <- read.csv(sprintf("%s/%s_fold%d_train.csv",base,model,fold),check.names=FALSE)
      va <- read.csv(sprintf("%s/%s_fold%d_valid.csv",base,model,fold),check.names=FALSE)
      x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
      v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
      y <- tr$y; z <- va$y
      if(sum(y)<5 || length(unique(y))<2) stop("inner class support insufficient")
      fit <- glmnet(x,y,family="binomial",alpha=alpha,lambda=grid,
                    standardize=FALSE,intercept=TRUE,thresh=1e-8,maxit=100000)
      if(length(fit$lambda)!=30) stop("explicit lambda path truncated")
      pred <- predict(fit,newx=v,s=grid,type="response")
      p0 <- pmin(pmax((sum(y)+0.5)/(length(y)+1),1e-6),1-1e-6)
      ll0 <- -mean(z*log(p0)+(1-z)*log1p(-p0))
      for(i in seq_along(grid)) {
        p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
        ll <- -mean(z*log(p)+(1-z)*log1p(-p))
        records[[length(records)+1]] <- data.frame(model=model,alpha=alpha,
          lambda=grid[i],candidate_index=i,inner_fold=fold,
          skill=1-ll/ll0,n_train=length(y),n_valid=length(z))
      }
    }
  }
}
out <- do.call(rbind,records)
if(nrow(out)!=720) stop("expected 2 models x 3 alpha x 30 lambda x 4 folds")
key <- aggregate(skill~model+alpha+candidate_index+lambda,out,mean)
key <- key[order(key$model,-key$skill,key$alpha,key$candidate_index),]
winners <- do.call(rbind,lapply(split(key,key$model),function(x) x[1,,drop=FALSE]))
write.csv(out,paste0(base,"/native_glmnet_shared_grid_trace.csv"),row.names=FALSE)
write.csv(winners,paste0(base,"/native_glmnet_shared_grid_winners.csv"),row.names=FALSE)
cat("PASS shared candidate grid, 720 inner candidate-fold scores; NO outer predictions\n")
print(winners)

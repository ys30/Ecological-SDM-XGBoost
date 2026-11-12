suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_r_native"
records <- list()
for (model in c("glmnet_lin","glmnet_quad_rt")) {
  for (fold in 0:3) {
    tr <- read.csv(sprintf("%s/%s_fold%d_train.csv",base,model,fold),check.names=FALSE)
    va <- read.csv(sprintf("%s/%s_fold%d_valid.csv",base,model,fold),check.names=FALSE)
    x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
    v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
    y <- tr$y; z <- va$y
    if (sum(y)<5 || length(unique(y))<2) stop("class support insufficient: record fallback separately")
    nullp <- (sum(y)+0.5)/(length(y)+1)
    nullp <- pmin(pmax(nullp,1e-6),1-1e-6)
    null_ll <- -mean(z*log(nullp)+(1-z)*log1p(-nullp))
    for (alpha in c(0,0.5,1)) {
      fit <- glmnet(x,y,family="binomial",alpha=alpha,nlambda=30,
                    standardize=FALSE,intercept=TRUE,thresh=1e-8,maxit=100000)
      if (length(fit$lambda)<2) stop("glmnet returned degenerate path")
      pred <- predict(fit,newx=v,type="response")
      for (i in seq_along(fit$lambda)) {
        p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
        ll <- -mean(z*log(p)+(1-z)*log1p(-p))
        records[[length(records)+1]] <- data.frame(model=model,inner_fold=fold,
          alpha=alpha,lambda=fit$lambda[i],path_index=i,
          path_length=length(fit$lambda),skill=1-ll/null_ll,
          n_train=length(y),n_valid=length(z))
      }
    }
  }
}
out <- do.call(rbind,records)
write.csv(out,paste0(base,"/native_glmnet_path_trace.csv"),row.names=FALSE)
cat("PASS native glmnet training-only paths; total candidate-fold rows:",nrow(out),"\n")
print(aggregate(skill~model+alpha,out,max))
cat("NOTE: fold-specific lambda paths are not yet a comparable joint tuning grid; NO outer OOF produced.\n")

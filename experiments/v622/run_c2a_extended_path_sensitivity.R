# C2a lambda-boundary sensitivity, DIAGNOSTIC ONLY.
# Extend each INNER-TRAINING glmnet path one decade below its observed minimum.
# Do not use validation outcomes to determine lambda candidates.
suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_outer_smoke"
rows <- list()
for(model in c("glmnet_lin","glmnet_quad_rt")) {
  trace <- list()
  for(alpha in c(0,0.5,1)) {
    for(f in 0:3) {
      tr <- read.csv(sprintf("%s/%s_inner%d_train.csv",base,model,f),check.names=FALSE)
      va <- read.csv(sprintf("%s/%s_inner%d_valid.csv",base,model,f),check.names=FALSE)
      y <- tr$y;z <- va$y
      x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
      v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
      fallback <- sum(y)<5 || length(unique(y))<2
      auto <- if(fallback) NULL else tryCatch(glmnet(x,y,family="binomial",
             alpha=alpha,nlambda=100,standardize=FALSE,thresh=1e-8,
             maxit=100000),error=function(e)NULL)
      if(is.null(auto) || length(auto$lambda)<2) {
        fallback <- TRUE
        grid <- rep(NA_real_,30)
        pred <- matrix(mean(y),nrow=nrow(va),ncol=30)
      } else {
        grid <- exp(seq(log(max(auto$lambda)),log(min(auto$lambda)/10),length.out=30))
        fit <- tryCatch(glmnet(x,y,family="binomial",alpha=alpha,lambda=grid,
                  standardize=FALSE,thresh=1e-8,maxit=100000),error=function(e)NULL)
        if(is.null(fit)) {
          fallback <- TRUE
          pred <- matrix(mean(y),nrow=nrow(va),ncol=30)
        } else pred <- predict(fit,newx=v,s=grid,type="response")
      }
      p0 <- (sum(y)+0.5)/(length(y)+1)
      ll0 <- -mean(z*log(p0)+(1-z)*log1p(-p0))
      for(i in seq_len(30)) {
        p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
        ll <- -mean(z*log(p)+(1-z)*log1p(-p))
        trace[[length(trace)+1]] <- data.frame(model=model,alpha=alpha,index=i,
            progress=(i-1)/29,inner_fold=f,actual_lambda=grid[i],
            skill=1-ll/ll0,fallback=fallback)
      }
    }
  }
  t <- do.call(rbind,trace)
  if(nrow(t)!=360) stop("wrong trace size")
  agg <- aggregate(skill~alpha+index+progress,t,mean)
  best <- agg[order(-agg$skill,agg$alpha,agg$index),][1,]
  tr <- read.csv(sprintf("%s/%s_outer_train.csv",base,model),check.names=FALSE)
  te <- read.csv(sprintf("%s/%s_outer_test.csv",base,model),check.names=FALSE)
  y <- tr$y
  x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
  v <- as.matrix(te[,setdiff(names(te),c("y","plot")),drop=FALSE])
  auto <- glmnet(x,y,family="binomial",alpha=best$alpha,nlambda=100,
                 standardize=FALSE,thresh=1e-8,maxit=100000)
  grid <- exp(seq(log(max(auto$lambda)),log(min(auto$lambda)/10),length.out=30))
  fit <- glmnet(x,y,family="binomial",alpha=best$alpha,lambda=grid,
                standardize=FALSE,thresh=1e-8,maxit=100000)
  chosen <- grid[best$index]
  pred <- as.numeric(predict(fit,newx=v,s=chosen,type="response"))
  if(length(pred)!=nrow(te) || any(!is.finite(pred)) ||
     any(pred<0|pred>1)) stop("invalid outer predictions")
  rows[[model]] <- data.frame(plot=te$plot,Species="nsw18",outer_fold=1,
     y=te$y,pred=pred,model_id=model,alpha=best$alpha,
     lambda=chosen,lambda_progress=best$progress,
     fallback=FALSE,diagnostic_only=TRUE)
  write.csv(t,sprintf("%s/%s_extended_path_trace.csv",base,model),row.names=FALSE)
  cat("C2A EXTENDED PATH",model,"alpha",best$alpha,"progress",best$progress,
      "lambda",chosen,"inner_skill",best$skill,"\n")
}
out <- do.call(rbind,rows)
write.csv(out,paste0(base,"/outer_extended_path_DIAGNOSTIC_ONLY.csv"),row.names=FALSE)
cat("PASS C2a extended path sensitivity",nrow(out),"diagnostic OOF rows\n")

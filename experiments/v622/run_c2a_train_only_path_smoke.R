# C2a leakage-safe native glmnet path smoke, diagnostic only.
# Each inner glmnet path uses ONLY its own training labels.
# Compare path positions by normalized log-lambda progress, never numeric lambda across folds.
suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_outer_smoke"
progress <- seq(0,1,length.out=30)
rows <- list()
for(model in c("glmnet_lin","glmnet_quad_rt")) {
  scores <- list()
  for(alpha in c(0,0.5,1)) {
    for(f in 0:3) {
      tr <- read.csv(sprintf("%s/%s_inner%d_train.csv",base,model,f),check.names=FALSE)
      va <- read.csv(sprintf("%s/%s_inner%d_valid.csv",base,model,f),check.names=FALSE)
      y <- tr$y; z <- va$y
      x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
      v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
      fallback <- sum(y)<5 || length(unique(y))<2
      fit <- if(fallback) NULL else tryCatch(glmnet(x,y,family="binomial",
          alpha=alpha,nlambda=100,standardize=FALSE,intercept=TRUE,
          thresh=1e-8,maxit=100000),error=function(e)NULL)
      if(is.null(fit) || length(fit$lambda)<2) {
        pred <- matrix(mean(y),nrow=nrow(va),ncol=30)
        actual_lambda <- rep(NA_real_,30)
        fallback <- TRUE
      } else {
        # Endpoints of the training-only path, with 30 equally spaced log positions.
        actual_lambda <- exp(log(max(fit$lambda))+
                       progress*(log(min(fit$lambda))-log(max(fit$lambda))))
        pred <- predict(fit,newx=v,s=actual_lambda,type="response")
      }
      p0 <- (sum(y)+0.5)/(length(y)+1)
      ll0 <- -mean(z*log(p0)+(1-z)*log1p(-p0))
      for(i in seq_along(progress)) {
        p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
        ll <- -mean(z*log(p)+(1-z)*log1p(-p))
        scores[[length(scores)+1]] <- data.frame(model=model,alpha=alpha,
             index=i,progress=progress[i],inner_fold=f,
             actual_lambda=actual_lambda[i],skill=1-ll/ll0,fallback=fallback)
      }
    }
  }
  trace <- do.call(rbind,scores)
  if(nrow(trace)!=360) stop("incorrect inner trace count")
  agg <- aggregate(skill~alpha+index+progress,trace,mean)
  best <- agg[order(-agg$skill,agg$alpha,agg$index),][1,]
  tr <- read.csv(sprintf("%s/%s_outer_train.csv",base,model),check.names=FALSE)
  te <- read.csv(sprintf("%s/%s_outer_test.csv",base,model),check.names=FALSE)
  y <- tr$y
  x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
  v <- as.matrix(te[,setdiff(names(te),c("y","plot")),drop=FALSE])
  outer_fallback <- sum(y)<5 || length(unique(y))<2
  fit <- if(outer_fallback) NULL else tryCatch(glmnet(x,y,family="binomial",
      alpha=best$alpha,nlambda=100,standardize=FALSE,intercept=TRUE,
      thresh=1e-8,maxit=100000),error=function(e)NULL)
  if(is.null(fit) || length(fit$lambda)<2) {
    pred <- rep(mean(y),nrow(te)); chosen_lambda <- NA_real_
    outer_fallback <- TRUE
  } else {
    chosen_lambda <- exp(log(max(fit$lambda))+
           best$progress*(log(min(fit$lambda))-log(max(fit$lambda))))
    pred <- as.numeric(predict(fit,newx=v,s=chosen_lambda,type="response"))
  }
  if(length(pred)!=nrow(te) || any(!is.finite(pred)) ||
     any(pred<0|pred>1)) stop("invalid predictions")
  rows[[model]] <- data.frame(plot=te$plot,Species="nsw18",outer_fold=1,
      y=te$y,pred=pred,model_id=model,alpha=best$alpha,
      lambda=chosen_lambda,lambda_progress=best$progress,
      fallback=outer_fallback,diagnostic_only=TRUE)
  write.csv(trace,sprintf("%s/%s_train_only_path_trace.csv",base,model),row.names=FALSE)
  cat("C2A TRAIN-ONLY PATH",model,"alpha",best$alpha,
      "progress",best$progress,"lambda",chosen_lambda,
      "skill",best$skill,"fallback",outer_fallback,"\n")
}
out <- do.call(rbind,rows)
if(nrow(out)!=474) stop("unexpected OOF count")
write.csv(out,paste0(base,"/outer_train_only_path_DIAGNOSTIC_ONLY.csv"),row.names=FALSE)
cat("PASS train-only glmnet lambda-path diagnostic",nrow(out),"rows; NOT formal C2a\n")

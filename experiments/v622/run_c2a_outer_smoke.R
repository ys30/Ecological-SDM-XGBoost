# C2a OUTER OOF SMOKE ONLY: nsw18/grid4/fold1.
# Fixed lambda grid is an explicitly labeled diagnostic deviation from
# preregistered automatic glmnet paths, not confirmatory model selection.
suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_outer_smoke"
grid <- exp(seq(log(10),log(1e-6),length.out=30))
records <- list()
for(model in c("glmnet_lin","glmnet_quad_rt")) {
  scores <- data.frame()
  for(alpha in c(0,0.5,1)) {
    for(f in 0:3) {
      tr <- read.csv(sprintf("%s/%s_inner%d_train.csv",base,model,f),check.names=FALSE)
      va <- read.csv(sprintf("%s/%s_inner%d_valid.csv",base,model,f),check.names=FALSE)
      y <- tr$y; z <- va$y
      x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
      v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
      fallback <- sum(y)<5 || length(unique(y))<2
      pred <- if(fallback) matrix(mean(y),nrow=nrow(va),ncol=30) else {
        fit <- tryCatch(glmnet(x,y,family="binomial",alpha=alpha,lambda=grid,
                     standardize=FALSE,intercept=TRUE,maxit=100000),error=function(e)NULL)
        if(is.null(fit)) matrix(mean(y),nrow=nrow(va),ncol=30)
        else predict(fit,newx=v,s=grid,type="response")
      }
      p0 <- (sum(y)+0.5)/(length(y)+1)
      ll0 <- -mean(z*log(p0)+(1-z)*log1p(-p0))
      for(i in seq_along(grid)) {
        p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
        ll <- -mean(z*log(p)+(1-z)*log1p(-p))
        scores <- rbind(scores,data.frame(alpha=alpha,index=i,lambda=grid[i],
                     fold=f,skill=1-ll/ll0,fallback=fallback))
      }
    }
  }
  agg <- aggregate(skill~alpha+index+lambda,scores,mean)
  best <- agg[order(-agg$skill,agg$alpha,agg$index),][1,]
  tr <- read.csv(sprintf("%s/%s_outer_train.csv",base,model),check.names=FALSE)
  te <- read.csv(sprintf("%s/%s_outer_test.csv",base,model),check.names=FALSE)
  y <- tr$y
  x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
  v <- as.matrix(te[,setdiff(names(te),c("y","plot")),drop=FALSE])
  fallback <- sum(y)<5 || length(unique(y))<2
  pred <- if(fallback) rep(mean(y),nrow(te)) else {
    fit <- glmnet(x,y,family="binomial",alpha=best$alpha,lambda=best$lambda,
                  standardize=FALSE,intercept=TRUE,maxit=100000)
    as.numeric(predict(fit,newx=v,s=best$lambda,type="response"))
  }
  if(length(pred)!=nrow(te) || any(!is.finite(pred)) || any(pred<0|pred>1))
    stop("invalid outer predictions")
  records[[model]] <- data.frame(plot=te$plot,Species="nsw18",outer_fold=1,
      y=te$y,pred=pred,model_id=model,alpha=best$alpha,lambda=best$lambda,
      fallback=fallback,diagnostic_only=TRUE)
  write.csv(scores,sprintf("%s/%s_inner_trace.csv",base,model),row.names=FALSE)
  cat("C2A OUTER SMOKE",model,"n_test",length(pred),"alpha",best$alpha,
      "lambda",best$lambda,"inner_skill",best$skill,"fallback",fallback,"\n")
}
out <- do.call(rbind,records)
if(nrow(out)!=2*nrow(te)) stop("wrong smoke output count")
write.csv(out,paste0(base,"/outer_smoke_oof_DIAGNOSTIC_ONLY.csv"),row.names=FALSE)
cat("PASS C2a diagnostic outer OOF",nrow(out),"rows; NOT formal C2a inference\n")

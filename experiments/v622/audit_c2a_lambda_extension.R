suppressPackageStartupMessages(library(glmnet))
base <- "results/v622/c2a_r_native"
model <- "glmnet_quad_rt"
alpha <- 0.5
traces <- read.csv(paste0(base,"/native_glmnet_shared_grid_trace.csv"))
old <- subset(traces,model==model & alpha==0.5)
# Use the exact baseline selected lambda; sensitivity is NOT a primary grid change.
lambda0 <- 8.59802678336364e-05
multipliers <- c(1,0.5,0.25,0.1,0.05,0.01)
results <- list()
for (fold in 0:3) {
  tr <- read.csv(sprintf("%s/%s_fold%d_train.csv",base,model,fold),check.names=FALSE)
  va <- read.csv(sprintf("%s/%s_fold%d_valid.csv",base,model,fold),check.names=FALSE)
  x <- as.matrix(tr[,setdiff(names(tr),"y"),drop=FALSE])
  v <- as.matrix(va[,setdiff(names(va),"y"),drop=FALSE])
  y <- tr$y; z <- va$y
  path <- sort(lambda0*multipliers,decreasing=TRUE)
  fit <- glmnet(x,y,family="binomial",alpha=alpha,lambda=path,
                standardize=FALSE,intercept=TRUE,thresh=1e-8,maxit=100000)
  pred <- predict(fit,newx=v,s=path,type="response")
  p0 <- pmin(pmax((sum(y)+0.5)/(length(y)+1),1e-6),1-1e-6)
  ll0 <- -mean(z*log(p0)+(1-z)*log1p(-p0))
  for (i in seq_along(path)) {
    p <- pmin(pmax(as.numeric(pred[,i]),1e-6),1-1e-6)
    ll <- -mean(z*log(p)+(1-z)*log1p(-p))
    results[[length(results)+1]] <- data.frame(model=model,alpha=alpha,
      inner_fold=fold,lambda=path[i],lambda_multiplier=path[i]/lambda0,
      skill=1-ll/ll0,n_train=length(y),n_valid=length(z))
  }
}
d <- do.call(rbind,results)
if(nrow(d)!=24) stop("expected 4 folds x 6 lambda sensitivity candidates")
write.csv(d,paste0(base,"/quad_rt_boundary_extension_trace.csv"),row.names=FALSE)
summary <- aggregate(skill~lambda_multiplier+lambda,d,mean)
summary <- summary[order(-summary$lambda_multiplier),]
write.csv(summary,paste0(base,"/quad_rt_boundary_extension_summary.csv"),row.names=FALSE)
print(summary)
cat("PASS 24 sensitivity-only scores; no outer OOF; baseline path unchanged.\n")

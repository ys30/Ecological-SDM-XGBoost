# Bayesian extension experiment: exploratory quadratic vs no-pooling comparisons.
# Five archived plot-grouped spatial folds; no held-out data in preprocessing.
suppressPackageStartupMessages({library(brms);library(pROC)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2)
variant <- args[1]; fold <- as.integer(args[2])
stopifnot(variant %in% c("quadratic","no_pool"),fold %in% 1:5)
d <- read.csv("data/train.csv",check.names=FALSE)
a <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- a$fold[match(d$plot,a$plot)]
stopifnot(!anyNA(d$fold),!anyDuplicated(d$id))
d$species <- factor(d$Species)
tr <- d[d$fold!=fold,,drop=FALSE]; te <- d[d$fold==fold,,drop=FALSE]
stopifnot(length(intersect(unique(tr$plot),unique(te$plot)))==0)
for(nm in c("rainann","tempann","soildepth","topo")) {
  mu <- mean(tr[[nm]]); sig <- sd(tr[[nm]])
  stopifnot(is.finite(mu),is.finite(sig),sig>0)
  tr[[paste0("z_",nm)]] <- (tr[[nm]]-mu)/sig
  te[[paste0("z_",nm)]] <- (te[[nm]]-mu)/sig
}
if(variant=="quadratic") {
  f <- bf(pres.abs ~ z_rainann + z_tempann + z_soildepth + z_topo +
          I(z_rainann^2) + I(z_tempann^2) +
          (1 + z_rainann + z_tempann || species))
} else {
  # No pooling across species: separate intercepts and slopes; regularizing priors.
  # Same four predictors as the archived hierarchical baseline.
  f <- bf(pres.abs ~ 0 + species + species:z_rainann +
          species:z_tempann + species:z_soildepth + species:z_topo)
}
pri <- if (variant == "quadratic") {
  c(prior(normal(-2.5,2),class="Intercept"), prior(normal(0,1),class="b"), prior(exponential(2),class="sd"))
} else {
  c(prior(normal(0,2),class="b"))
}
fit <- brm(f,data=tr,family=bernoulli(),prior=pri,
           chains=4,iter=2000,warmup=1000,cores=4,seed=3026+fold,
           refresh=0,control=list(adapt_delta=0.97))
dir.create("results/bayes_extensions",recursive=TRUE,showWarnings=FALSE)
diag <- posterior::summarise_draws(posterior::as_draws_array(fit),"rhat","ess_bulk","ess_tail")
write.csv(as.data.frame(diag),sprintf("results/bayes_extensions/%s_fold%d_diagnostics.csv",variant,fold),row.names=FALSE)
sampler <- rstan::get_sampler_params(fit$fit,inc_warmup=FALSE)
sam <- do.call(rbind,lapply(seq_along(sampler),function(i) {
 x<-sampler[[i]]; e<-x[,"energy__"]
 data.frame(chain=i,divergences=sum(x[,"divergent__"]),
            treedepth_limit_hits=sum(x[,"treedepth__"]>=10),
            bfmi=mean(diff(e)^2)/var(e))
}))
write.csv(sam,sprintf("results/bayes_extensions/%s_fold%d_sampler.csv",variant,fold),row.names=FALSE)
p <- colMeans(posterior_epred(fit,newdata=te))
stopifnot(length(p)==nrow(te),all(is.finite(p)),all(p>=0&p<=1))
pred <- data.frame(id=te$id,plot=te$plot,species=te$Species,fold=fold,
                   observed=te$pres.abs,probability=p,variant=variant)
write.csv(pred,sprintf("results/bayes_extensions/%s_fold%d_oof.csv",variant,fold),row.names=FALSE)
metrics <- do.call(rbind,lapply(split(pred,pred$species),function(x){
 y<-x$observed; q<-x$probability
 auc<-if(length(unique(y))==2) as.numeric(pROC::auc(pROC::roc(y,q,quiet=TRUE,direction="<"))) else NA_real_
 ap<-if(sum(y)>0){o<-order(-q); yy<-y[o]==1;sum((cumsum(yy)/seq_along(yy))[yy])/sum(yy)}else NA_real_
 data.frame(variant=variant,fold=fold,species=x$species[1],n=nrow(x),
            positives=sum(y),auc=auc,ap=ap,brier=mean((y-q)^2),
            logloss=-mean(y*log(pmax(q,1e-12))+(1-y)*log(pmax(1-q,1e-12))))
}))
write.csv(metrics,sprintf("results/bayes_extensions/%s_fold%d_metrics.csv",variant,fold),row.names=FALSE)
stopifnot(all(is.finite(diag$rhat)),max(diag$rhat)<=1.01,
          min(diag$ess_bulk)>=400,min(diag$ess_tail)>=400,
          all(sam$divergences==0),all(sam$treedepth_limit_hits==0),all(sam$bfmi>=0.3))
cat("PASS",variant,"fold",fold,"\n")

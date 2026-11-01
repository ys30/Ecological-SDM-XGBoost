# Bayesian partial-pooling species distribution model.
# Five pre-existing spatial folds; held-out blocks never enter training.
suppressPackageStartupMessages({library(brms);library(pROC)})
d <- read.csv("data/train.csv",check.names=FALSE)
a <- read.csv("results/spatial_2d_block_assignments.csv")
d$fold <- a$fold[match(d$plot,a$plot)]
stopifnot(!anyNA(d$fold),!anyDuplicated(d$id))
d$species <- factor(d$Species)
# Predictor scaling is recomputed from training data for every fold below.
formula <- bf(pres.abs ~ z_rainann + z_tempann + z_soildepth + z_topo +
                 (1 + z_rainann + z_tempann || species))
priors <- c(prior(normal(-2.5,2),class="Intercept"),
            prior(normal(0,1),class="b"),
            prior(exponential(2),class="sd"))
dir.create("results",showWarnings=FALSE)
args <- commandArgs(trailingOnly=TRUE)
folds <- if(length(args)) as.integer(args[!grepl("^--",args)]) else 1:5
confirm <- "--confirm" %in% args
nchains <- if(confirm) 4L else 2L
niter <- if(confirm) 2000L else 1200L
nwarmup <- if(confirm) 1000L else 600L
suffix <- if(confirm) "_confirm" else ""
stopifnot(length(folds)>0,all(!is.na(folds)),all(folds %in% 1:5))
metrics <- list(); predictions <- list()
for(k in folds) {
 tr <- d[d$fold!=k,,drop=FALSE]
 te <- d[d$fold==k,,drop=FALSE]
 stopifnot(!length(intersect(unique(tr$plot),unique(te$plot))))
 for(nm in c("rainann","tempann","soildepth","topo")) {
   mu <- mean(tr[[nm]]); sig <- sd(tr[[nm]])
   stopifnot(is.finite(sig),sig>0)
   tr[[paste0("z_",nm)]] <- (tr[[nm]]-mu)/sig
   te[[paste0("z_",nm)]] <- (te[[nm]]-mu)/sig
 }
 fit <- brm(formula,data=tr,family=bernoulli(),prior=priors,
   chains=nchains,iter=niter,warmup=nwarmup,cores=nchains,seed=2026+k,
   refresh=0,control=list(adapt_delta=0.95))
 diag <- posterior::summarise_draws(posterior::as_draws_array(fit), "rhat", "ess_bulk", "ess_tail")
 # Stan sampler diagnostics are necessary beyond R-hat and effective sample size.
 sampler <- rstan::get_sampler_params(fit$fit,inc_warmup=FALSE)
 stopifnot(length(sampler)==nchains)
 chain_diag <- lapply(seq_along(sampler),function(i) {
   x <- sampler[[i]]
   stopifnot(all(c("divergent__","treedepth__","energy__") %in% colnames(x)))
   e <- x[,"energy__"]
   bfmi <- mean(diff(e)^2)/stats::var(e)
   stopifnot(is.finite(bfmi))
   data.frame(chain=i,post_warmup_draws=nrow(x),divergences=sum(x[,"divergent__"]),
     max_treedepth_observed=max(x[,"treedepth__"]),
     treedepth_at_limit=sum(x[,"treedepth__"]>=10),bfmi=bfmi)
 })
 write.csv(do.call(rbind,chain_diag),sprintf("results/bayes_spatial_fold_%d_sampler_diagnostics%s.csv",k,suffix),row.names=FALSE)
 write.csv(as.data.frame(diag),sprintf("results/bayes_spatial_fold_%d_diagnostics%s.csv",k,suffix),row.names=FALSE)
 saveRDS(fit,sprintf("results/bayes_spatial_fold_%d%s.rds",k,suffix))
 pdraw <- posterior_epred(fit,newdata=te,allow_new_levels=TRUE)
 p <- colMeans(pdraw)
 auc <- if(length(unique(te$pres.abs))==2)
   as.numeric(pROC::auc(pROC::roc(te$pres.abs,p,quiet=TRUE,direction="<"))) else NA_real_
 ap <- if(sum(te$pres.abs)>0) {
   o <- order(-p); yy <- te$pres.abs[o]==1
   sum((cumsum(yy)/seq_along(yy))[yy])/sum(yy)
 } else NA_real_
 metrics[[k]] <- data.frame(fold=k,n=nrow(te),auc=auc,ap=ap,
   logloss=-mean(te$pres.abs*log(pmax(p,1e-12))+
      (1-te$pres.abs)*log(pmax(1-p,1e-12))))
 predictions[[k]] <- data.frame(id=te$id,plot=te$plot,species=te$Species,
   fold=k,observed=te$pres.abs,mean_probability=p,
   lower_95=apply(pdraw,2,quantile,.025),
   upper_95=apply(pdraw,2,quantile,.975))
}
write.csv(do.call(rbind,metrics),sprintf("results/bayes_spatial_cv_metrics_folds_%s%s.csv",paste(folds,collapse="-"),suffix),row.names=FALSE)
write.csv(do.call(rbind,predictions),sprintf("results/bayes_spatial_cv_predictions_folds_%s%s.csv",paste(folds,collapse="-"),suffix),row.names=FALSE)

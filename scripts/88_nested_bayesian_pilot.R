suppressPackageStartupMessages({library(brms);library(posterior)})
set.seed(3029)
d <- read.csv("data/train.csv",check.names=FALSE)
roles <- read.csv("results/nested_bayesian_pilot/fold3_10km_roles.csv")
d$species <- factor(d$Species)
out <- "results/nested_bayesian_pilot"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
all_inner <- list()
for(k in c(0,1,2,3,-1)) {
 rr <- roles[roles$inner_fold==k,]
 tr <- d[d$plot %in% rr$plot[rr$role=="train"],,drop=FALSE]
 te <- d[d$plot %in% rr$plot[rr$role==if(k==-1) "test" else "validation"],,drop=FALSE]
 stopifnot(nrow(tr)>0,nrow(te)>0,length(intersect(unique(tr$plot),unique(te$plot)))==0)
 for(nm in c("rainann","tempann","soildepth","topo")) {
  mu<-mean(tr[[nm]]); sig<-sd(tr[[nm]]);stopifnot(is.finite(sig),sig>0)
  tr[[paste0("z_",nm)]]<-(tr[[nm]]-mu)/sig
  te[[paste0("z_",nm)]]<-(te[[nm]]-mu)/sig
 }
 form<-bf(pres.abs ~ z_rainann+z_tempann+z_soildepth+z_topo+
   I(z_rainann^2)+I(z_tempann^2)+(1+z_rainann+z_tempann||species))
 pri<-c(prior(normal(-2.5,2),class="Intercept"),prior(normal(0,1),class="b"),prior(exponential(2),class="sd"))
 fit<-brm(form,data=tr,family=bernoulli(),prior=pri,
   chains=4,iter=2000,warmup=1000,cores=4,seed=3026+3+ifelse(k==-1,0,k+1),
   refresh=0,control=list(adapt_delta=0.97))
 diag<-posterior::summarise_draws(posterior::as_draws_array(fit),"rhat","ess_bulk","ess_tail")
 sampler<-rstan::get_sampler_params(fit$fit,inc_warmup=FALSE)
 div<-sum(vapply(sampler,function(x)sum(x[,"divergent__"]),numeric(1)))
 stopifnot(all(is.finite(diag$rhat)),max(diag$rhat)<=1.01,
           min(diag$ess_bulk)>=400,min(diag$ess_tail)>=400,div==0)
 pp<-colMeans(posterior_epred(fit,newdata=te))
 result<-data.frame(species=te$Species,observed=te$pres.abs,prob=pp)
 if(k==-1) {
   write.csv(result,file.path(out,"outer_predictions_PRIVATE_CI_ONLY.csv"),row.names=FALSE)
 } else {
   all_inner[[as.character(k)]]<-result
 }
 cat("PASS fit",k,"train plots",length(unique(tr$plot)),"validation plots",length(unique(te$plot)),"\n")
}
inner<-do.call(rbind,all_inner)
outer<-read.csv(file.path(out,"outer_predictions_PRIVATE_CI_ONLY.csv"))
rows<-list()
for(taxon in c("nsw19","nsw20","nsw22")) {
 neg<-inner$prob[inner$species==taxon & inner$observed==0]
 oo<-outer[outer$species==taxon,]
 for(rate in c(.01,.02,.05)) {
  allowed<-floor(rate*length(neg))
  threshold<-sort(neg,decreasing=TRUE)[allowed+1]
  threshold<-threshold + .Machine$double.eps*max(1,abs(threshold))
  pred<-oo$prob>=threshold
  rows[[length(rows)+1]]<-data.frame(species=taxon,rate=rate,threshold=threshold,
    tp=sum(pred & oo$observed==1),fp=sum(pred & oo$observed==0),
    fn=sum(!pred & oo$observed==1),tn=sum(!pred & oo$observed==0),
    inner_n_neg=length(neg),inner_fp=sum(neg>=threshold))
 }
}
write.csv(do.call(rbind,rows),file.path(out,"fold3_10km_nested_bayesian_counts.csv"),row.names=FALSE)
unlink(file.path(out,"outer_predictions_PRIVATE_CI_ONLY.csv"))
cat("NESTED BAYES PILOT\n")
print(do.call(rbind,rows))

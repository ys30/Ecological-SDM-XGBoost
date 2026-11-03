# EMS v6.14 audit: matched OOF predictions, leakage-free prevalence baseline
suppressPackageStartupMessages(library(pROC))
d <- read.csv("data/train.csv")
p <- read.csv("results/v614_common_input/taxon_specific_xgboost_oof.csv")
a <- read.csv("results/matched_bayesian_maxent_xgboost_folds.csv")
stopifnot(!anyDuplicated(p$id), nrow(p)==nrow(d), setequal(p$id,d$id),
          all(p$fold %in% 1:5), all(p$status %in% c("ok","single_class_training")))
stopifnot(all(is.finite(p$p[p$status=="ok"])),
          all(is.na(p$p[p$status!="ok"])),
          all(p$p[is.finite(p$p)]>0 & p$p[is.finite(p$p)]<1))
auc <- function(y,pr) if(length(unique(y))==2)
  as.numeric(pROC::auc(pROC::roc(y,pr,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
ap <- function(y,pr) {
 if(sum(y)==0)return(NA_real_)
 o<-order(pr,decreasing=TRUE);z<-y[o]
 sum((cumsum(z)/seq_along(z))[z==1])/sum(z)
}
rows <- list()
for(sp in sort(unique(p$Species)))for(k in 1:5) {
 z<-p[p$Species==sp & p$fold==k & is.finite(p$p),,drop=FALSE]
 if(!nrow(z))next
 tr<-d[d$Species==sp & !(d$plot %in% z$plot),,drop=FALSE]
 # Fold assignment is defined by held-out plot IDs; no held-out prevalence used.
 stopifnot(nrow(tr)>0)
 q<-mean(tr$pres.abs)
 q<-min(max(q,1e-15),1-1e-15)
 rows[[length(rows)+1L]]<-data.frame(species=sp,fold=k,n=nrow(z),positives=sum(z$y),
   specific_auc=auc(z$y,z$p),specific_ap=ap(z$y,z$p),
   specific_brier=mean((z$y-z$p)^2),
   training_prevalence=q,
   baseline_brier=mean((z$y-q)^2),
   specific_logloss=-mean(z$y*log(z$p)+(1-z$y)*log1p(-z$p)),
   baseline_logloss=-mean(z$y*log(q)+(1-z$y)*log1p(-q)))
}
m<-do.call(rbind,rows)
stopifnot(!is.null(m),!anyDuplicated(paste(m$species,m$fold)))
x<-merge(a,m,by=c("species","fold"),all.x=TRUE,sort=FALSE)
stopifnot(nrow(x)==nrow(a))
# Never compare metrics from different evaluation samples.
stopifnot(all(is.finite(x$n.x)), all(is.finite(x$positives.x)))
stopifnot(all(!is.na(x$n.y)), all(!is.na(x$positives.y)))
stopifnot(all(x$n.x==x$n.y), all(x$positives.x==x$positives.y))
x$delta_specific_minus_pooled_auc<-x$specific_auc-x$xgboost_auc
x$delta_specific_minus_maxent_auc<-x$specific_auc-x$maxent_auc
x$delta_specific_minus_bayes_auc<-x$specific_auc-x$bayesian_auc
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(x,"results/v614_common_input/matched_comparison_audit.csv",row.names=FALSE)
cat("OOF rows:",nrow(p),"missing:",sum(!is.finite(p$p)),"\n")
cat("Matched eligible AUC cells:",sum(is.finite(x$specific_auc)),"/",nrow(a),"\n")
for(nm in c("delta_specific_minus_pooled_auc","delta_specific_minus_maxent_auc",
            "delta_specific_minus_bayes_auc")) {
 cat(nm,":",mean(x[[nm]],na.rm=TRUE),"\n")
}

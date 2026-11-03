# Audit common-input taxon-specific OOF predictions against archived pooled benchmark
suppressPackageStartupMessages(library(pROC))
p <- read.csv("results/v614_common_input/taxon_specific_xgboost_oof.csv")
a <- read.csv("results/matched_bayesian_maxent_xgboost_folds.csv")
stopifnot(!anyDuplicated(p$id),nrow(p)==nrow(read.csv("data/train.csv")),all(p$fold %in% 1:5))
stopifnot(all(p$status=="ok" | is.na(p$p)),all(p$p[is.finite(p$p)]>0 & p$p[is.finite(p$p)]<1))
auc <- function(y,pr) if(length(unique(y))==2) as.numeric(pROC::auc(pROC::roc(y,pr,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
ap <- function(y,pr) {if(!sum(y))return(NA_real_); o<-order(pr,decreasing=TRUE); z<-y[o];sum((cumsum(z)/seq_along(z))[z==1])/sum(z)}
rows <- list()
for(sp in unique(p$Species))for(k in 1:5){
 z<-p[p$Species==sp & p$fold==k & is.finite(p$p),]
 if(!nrow(z))next
 rows[[length(rows)+1]]<-data.frame(species=sp,fold=k,n=nrow(z),positives=sum(z$y),
  specific_auc=auc(z$y,z$p),specific_ap=ap(z$y,z$p),
  specific_brier=mean((z$y-z$p)^2),
  prevalence_brier=mean((z$y-mean(z$y))^2))
}
m<-do.call(rbind,rows)
x<-merge(a,m,by=c("species","fold"),all.x=TRUE)
x$delta_specific_minus_pooled_auc<-x$specific_auc-x$xgboost_auc
x$delta_specific_minus_maxent_auc<-x$specific_auc-x$maxent_auc
x$delta_specific_minus_bayes_auc<-x$specific_auc-x$bayesian_auc
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(x,"results/v614_common_input/matched_comparison_audit.csv",row.names=FALSE)
cat("Total OOF rows:",nrow(p),"\nMissing predictions:",sum(!is.finite(p$p)),"\n")
cat("Matched eligible AUC cells:",sum(is.finite(x$specific_auc)),"/",nrow(a),"\n")
print(aggregate(cbind(delta_specific_minus_pooled_auc,delta_specific_minus_maxent_auc,delta_specific_minus_bayes_auc)~1,data=x,FUN=mean,na.rm=TRUE))

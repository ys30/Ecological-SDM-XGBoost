# Matched pooling ablation: pooled four-env+species versus taxon-specific four-env
suppressPackageStartupMessages(library(pROC))
a<-read.csv("results/v614_common_input/taxon_specific_xgboost_oof.csv")
b<-read.csv("results/v614_common_input/pooled_four_plus_species_xgboost_oof.csv")
stopifnot(!anyDuplicated(a$id),!anyDuplicated(b$id),setequal(a$id,b$id))
z<-merge(a,b,by="id",suffixes=c("_specific","_pooled"))
stopifnot(nrow(z)==nrow(a),all(z$Species_specific==z$Species_pooled),
          all(z$fold_specific==z$fold_pooled),all(z$y_specific==z$y_pooled))
auc<-function(y,p) if(length(unique(y))==2)
 as.numeric(pROC::auc(pROC::roc(y,p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
ap<-function(y,p) {
 if(sum(y)==0)return(NA_real_)
 o<-order(p,decreasing=TRUE);yy<-y[o]
 sum((cumsum(yy)/seq_along(yy))[yy==1])/sum(yy)
}
rows<-list()
for(sp in sort(unique(z$Species_specific)))for(k in 1:5){
 q<-z[z$Species_specific==sp & z$fold_specific==k,,drop=FALSE]
 if(!nrow(q))next
 valid<-all(is.finite(q$p_specific)) && all(is.finite(q$p_pooled))
 rows[[length(rows)+1L]]<-data.frame(species=sp,fold=k,n=nrow(q),
  positives=sum(q$y_specific),status=if(valid)"ok" else "missing_prediction",
  specific_auc=if(valid)auc(q$y_specific,q$p_specific) else NA_real_,
  pooled_auc=if(valid)auc(q$y_specific,q$p_pooled) else NA_real_,
  specific_ap=if(valid)ap(q$y_specific,q$p_specific) else NA_real_,
  pooled_ap=if(valid)ap(q$y_specific,q$p_pooled) else NA_real_,
  specific_brier=if(valid)mean((q$y_specific-q$p_specific)^2) else NA_real_,
  pooled_brier=if(valid)mean((q$y_specific-q$p_pooled)^2) else NA_real_)
}
r<-do.call(rbind,rows)
r$delta_pooled_minus_specific_auc<-r$pooled_auc-r$specific_auc
r$delta_pooled_minus_specific_ap<-r$pooled_ap-r$specific_ap
write.csv(r,"results/v614_common_input/pooled_vs_specific_four_input.csv",row.names=FALSE)
cat("Matched eligible AUC cells:",sum(is.finite(r$delta_pooled_minus_specific_auc)),"\n")
cat("Mean pooled - specific AUC:",mean(r$delta_pooled_minus_specific_auc,na.rm=TRUE),"\n")
cat("Mean pooled - specific AP:",mean(r$delta_pooled_minus_specific_ap,na.rm=TRUE),"\n")

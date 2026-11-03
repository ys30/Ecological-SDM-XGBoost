# Matched three-way XGBoost ablation: taxon-specific, pooled+species, species-blind
suppressPackageStartupMessages(library(pROC))
root <- "results/v614_common_input/"
a <- read.csv(paste0(root,"taxon_specific_xgboost_oof.csv"))
b <- read.csv(paste0(root,"pooled_four_plus_species_xgboost_oof.csv"))
c <- read.csv(paste0(root,"pooled_four_only_xgboost_oof.csv"))
for(v in list(a,b,c)) stopifnot(!anyDuplicated(v$id),nrow(v)>0)
stopifnot(setequal(a$id,b$id),setequal(a$id,c$id))
names(a)[names(a)=="p"] <- "p_specific"
names(b)[names(b)=="p"] <- "p_pooled"
names(c)[names(c)=="p"] <- "p_blind"
keys <- c("id","plot","Species","fold","y")
stopifnot(all(keys %in% names(a)),all(keys %in% names(b)),all(keys %in% names(c)))
z <- merge(merge(a[,c(keys,"p_specific")],b[,c(keys,"p_pooled")],
 by=keys,all=FALSE),c[,c(keys,"p_blind")],by=keys,all=FALSE)
stopifnot(nrow(z)==nrow(a),nrow(z)==nrow(b),nrow(z)==nrow(c))
auc <- function(y,p) if(length(unique(y))==2)
 as.numeric(pROC::auc(pROC::roc(y,p,levels=c(0,1),direction="<",quiet=TRUE))) else NA_real_
ap <- function(y,p) {
 if(sum(y)==0)return(NA_real_)
 # Deterministic order for tied probabilities; ties must be examined in interpretation.
 o <- order(-p,seq_along(p)); yy<-y[o]
 sum((cumsum(yy)/seq_along(yy))[yy==1])/sum(yy)
}
rows <- list()
for(sp in sort(unique(z$Species))) for(k in sort(unique(z$fold))) {
 q<-z[z$Species==sp & z$fold==k,,drop=FALSE]
 if(!nrow(q))next
 valid<-all(is.finite(q$p_specific))&&all(is.finite(q$p_pooled))&&
        all(is.finite(q$p_blind))
 res<-data.frame(species=sp,fold=k,n=nrow(q),positives=sum(q$y),
   status=if(valid)"ok" else "missing_prediction")
 for(model in c("specific","pooled","blind")) {
  pr<-q[[paste0("p_",model)]]
  res[[paste0(model,"_auc")]]<-if(valid)auc(q$y,pr) else NA_real_
  res[[paste0(model,"_ap")]]<-if(valid)ap(q$y,pr) else NA_real_
  res[[paste0(model,"_brier")]]<-if(valid)mean((q$y-pr)^2) else NA_real_
 }
 rows[[length(rows)+1L]]<-res
}
r<-do.call(rbind,rows)
for(metric in c("auc","ap","brier")) {
 r[[paste0("delta_pooled_minus_specific_",metric)]] <-
  r[[paste0("pooled_",metric)]]-r[[paste0("specific_",metric)]]
 r[[paste0("delta_blind_minus_pooled_",metric)]] <-
  r[[paste0("blind_",metric)]]-r[[paste0("pooled_",metric)]]
}
write.csv(r,paste0(root,"three_way_xgboost_ablation.csv"),row.names=FALSE)
# Preserve legacy filename for downstream readers; same rows with added blind-model columns.
write.csv(r,paste0(root,"pooled_vs_specific_four_input.csv"),row.names=FALSE)
cat("Three-way evaluated cells:",nrow(r),"\n")
cat("Three-way eligible AUC cells:",sum(is.finite(r$delta_pooled_minus_specific_auc)),"\n")
for(nm in grep("^delta_",names(r),value=TRUE))
 cat(nm,":",if(all(is.na(r[[nm]]))) "NA" else mean(r[[nm]],na.rm=TRUE),"\n")

# EMS v6.14: retrospective within-fold rare-taxon retrieval using OOF predictions
# No prospective claims; no held-out outcomes used for model selection.
p <- read.csv("results/v614_common_input/taxon_specific_xgboost_oof.csv")
d <- read.csv("data/train.csv")
stopifnot(nrow(p)==nrow(d),!anyDuplicated(p$id),all(p$y %in% c(0,1)))
rare <- c("nsw19","nsw20","nsw22")
budgets <- c(10L,25L,50L)
rows <- list()
for(sp in rare) for(k in 1:5) {
 z <- p[p$Species==sp & p$fold==k,,drop=FALSE]
 if(!nrow(z)) stop(paste("Missing taxon/fold",sp,k))
 n <- nrow(z); positives <- sum(z$y)
 for(b in budgets) {
   stopifnot(b<=n)
   if(any(!is.finite(z$p))) {
     rows[[length(rows)+1L]] <- data.frame(species=sp,fold=k,n=n,positives=positives,
       budget=b,captured=NA_integer_,recall=NA_real_,random_expected=positives*b/n,
       enrichment=NA_real_,status="missing_prediction")
     next
   }
   # Stable deterministic tie breaking; prediction ranking is strictly within fold.
   ord <- order(-z$p,z$plot,z$id)
   cap <- sum(z$y[ord[seq_len(b)]])
   expectation <- positives*b/n
   rows[[length(rows)+1L]] <- data.frame(species=sp,fold=k,n=n,positives=positives,
     budget=b,captured=cap,recall=if(positives>0)cap/positives else NA_real_,
     random_expected=expectation,
     enrichment=if(expectation>0)cap/expectation else NA_real_,status="ok")
 }
}
r <- do.call(rbind,rows)
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(r,"results/v614_common_input/rare_taxon_within_fold_retrieval.csv",row.names=FALSE)
s <- aggregate(cbind(captured,positives,random_expected)~species+budget,data=r,
               FUN=function(x)if(anyNA(x)) NA_real_ else sum(x))
s$recall <- with(s,ifelse(positives>0,captured/positives,NA_real_))
s$enrichment <- with(s,ifelse(random_expected>0,captured/random_expected,NA_real_))
write.csv(s,"results/v614_common_input/rare_taxon_retrieval_summary.csv",row.names=FALSE)
print(s)

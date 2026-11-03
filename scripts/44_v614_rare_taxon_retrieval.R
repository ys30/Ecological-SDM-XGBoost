# EMS v6.14: retrospective within-fold rare-taxon retrieval using OOF predictions
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
   expectation <- positives*b/n
   if(any(!is.finite(z$p))) {
     rows[[length(rows)+1L]] <- data.frame(species=sp,fold=k,n=n,positives=positives,
       budget=b,captured=NA_integer_,recall=NA_real_,random_expected=expectation,
       enrichment=NA_real_,status="missing_prediction")
     next
   }
   ord <- order(-z$p,z$plot,z$id)
   cap <- sum(z$y[ord[seq_len(b)]])
   rows[[length(rows)+1L]] <- data.frame(species=sp,fold=k,n=n,positives=positives,
     budget=b,captured=cap,recall=if(positives>0)cap/positives else NA_real_,
     random_expected=expectation,
     enrichment=if(expectation>0)cap/expectation else NA_real_,status="ok")
 }
}
r <- do.call(rbind,rows)
dir.create("results/v614_common_input",recursive=TRUE,showWarnings=FALSE)
write.csv(r,"results/v614_common_input/rare_taxon_within_fold_retrieval.csv",row.names=FALSE)
groups <- split(r,list(r$species,r$budget),drop=TRUE)
s <- do.call(rbind,lapply(groups,function(z) {
 stopifnot(nrow(z)==5)
 valid <- all(z$status=="ok")
 positives <- sum(z$positives)
 expected <- sum(z$random_expected)
 captured <- if(valid)sum(z$captured) else NA_real_
 data.frame(species=z$species[1],budget=z$budget[1],
   captured=captured,positives=positives,random_expected=expected,
   recall=if(valid && positives>0)captured/positives else NA_real_,
   enrichment=if(valid && expected>0)captured/expected else NA_real_,
   status=if(valid)"ok" else "incomplete_predictions")
}))
row.names(s)<-NULL
write.csv(s,"results/v614_common_input/rare_taxon_retrieval_summary.csv",row.names=FALSE)
print(s)

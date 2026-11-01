#!/usr/bin/env Rscript
# Independent, stringent site-and-label identity audit against disdat NSW.
suppressPackageStartupMessages(library(disdat))
local <- read.csv("data/train.csv", check.names=FALSE)
pa <- disdat::disPa("NSW","ot")
env <- disdat::disEnv("NSW","ot")
dir.create("results",showWarnings=FALSE)
out <- "results/disdat_nsw_site_label_identity.txt"
sink(out)
on.exit(sink(),add=TRUE)
cat("SOURCE IDENTITY AUDIT — site coordinates, environment and binary labels\n")
stopifnot(all(c("x","y","siteid") %in% names(pa)),
          all(c("x","y","siteid") %in% names(env)))
key <- function(x,y) paste(sprintf("%.4f",as.numeric(x)),sprintf("%.4f",as.numeric(y)),sep="|")
loc <- local[!duplicated(local$plot),,drop=FALSE]
lk <- key(loc$long,loc$lat); pk <- key(pa$x,pa$y); ek <- key(env$x,env$y)
cat("Local plots:",nrow(loc),"Reference PA sites:",nrow(pa),"Reference ENV sites:",nrow(env),"\n")
cat("Unique geographic keys local/PA/ENV:",length(unique(lk)),length(unique(pk)),length(unique(ek)),"\n")
cat("Local coordinate matches in PA:",sum(lk %in% pk),"of",length(lk),"\n")
cat("Local coordinate matches in ENV:",sum(lk %in% ek),"of",length(lk),"\n")
if(anyDuplicated(lk)||anyDuplicated(pk)||anyDuplicated(ek)) {
  cat("STOP: duplicate rounded geographic keys; cannot assert unambiguous site identity.\n")
} else if(!all(lk %in% pk) || !all(lk %in% ek)) {
  cat("STOP: incomplete geographic match.\n")
} else {
  ip <- match(lk,pk); ie <- match(lk,ek)
  shared <- intersect(c("disturb","rainann","soildepth","soilfert","tempann","topo"),names(env))
  match_env <- vapply(shared,function(z) all(as.character(loc[[z]])==as.character(env[[z]][ie])),logical(1))
  cat("Exact covariate equality by field:\n"); print(match_env)
  cat("Reference PA/ENV siteid consistent:",all(as.character(pa$siteid[ip])==as.character(env$siteid[ie])),"\n")
  codes <- grep("^nsw[0-9]+$",names(pa),value=TRUE)
  spp <- sort(unique(as.character(local$Species)))
  obs <- matrix(NA_integer_,nrow=nrow(loc),ncol=length(spp),dimnames=list(NULL,spp))
  for(j in seq_along(spp)) {
    d <- local[local$Species==spp[j],c("plot","pres.abs")]
    obs[,j] <- as.integer(d$pres.abs[match(loc$plot,d$plot)])
  }
  ref <- as.matrix(pa[ip,codes,drop=FALSE]); storage.mode(ref)<-"integer"
  mismatch <- outer(seq_along(spp),seq_along(codes),Vectorize(function(i,j)
    sum(obs[,i]!=ref[,j],na.rm=TRUE)))
  dimnames(mismatch)<-list(spp,codes)
  cat("Species-by-code mismatch matrix:\n"); print(mismatch)
  # Evaluate all 8! assignments, not just marginal positive totals.
  best <- Inf; best_perm <- NULL; ties <- 0L
  rec <- function(i,used,perm,total) {
    if(total>best)return(invisible(NULL))
    if(i>length(spp)) {
      if(total<best){best <<- total;best_perm <<- perm;ties <<- 1L}
      else if(total==best) ties <<- ties+1L
      return(invisible(NULL))
    }
    for(j in which(!used)) {
      used2<-used;used2[j]<-TRUE
      rec(i+1L,used2,c(perm,j),total+mismatch[i,j])
    }
  }
  rec(1L,rep(FALSE,length(codes)),integer(),0)
  cat("Minimum binary-label disagreements:",best,"out of",nrow(obs)*ncol(obs),"\n")
  cat("Number of optimal mappings:",ties,"\n")
  cat("Best mapping:\n")
  print(data.frame(species=spp,disdat_code=codes[best_perm],disagreements=mismatch[cbind(seq_along(spp),best_perm)]))
  cat("FULL_IDENTITY_CONFIRMED:",best==0 && ties==1 && all(match_env) &&
      all(as.character(pa$siteid[ip])==as.character(env$siteid[ie])),"\n")
}
sink()
cat(readLines(out),sep="\n")

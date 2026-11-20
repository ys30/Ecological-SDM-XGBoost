#!/usr/bin/env Rscript
# Independently replay the archived coordinate-only fold allocation.
# Uses base R only. No model fitting, labels, or changes to frozen files.
d <- read.csv("data/train.csv", stringsAsFactors=FALSE)
saved <- read.csv("results/spatial_2d_block_assignments.csv", stringsAsFactors=FALSE)
stopifnot(nrow(d)==9336L, nrow(saved)==1167L)
plots <- aggregate(cbind(long,lat) ~ plot, data=d, FUN=median)
qx <- unique(quantile(plots$long, probs=seq(0,1,length.out=5),names=FALSE))
qy <- unique(quantile(plots$lat, probs=seq(0,1,length.out=5),names=FALSE))
stopifnot(length(qx)>=3L,length(qy)>=3L)
plots$ix <- findInterval(plots$long,qx,all.inside=TRUE)
plots$iy <- findInterval(plots$lat,qy,all.inside=TRUE)
plots$block <- paste(plots$ix,plots$iy,sep="_")
blocks <- as.data.frame(table(plots$block),stringsAsFactors=FALSE)
names(blocks) <- c("block","nplots")
blocks <- blocks[blocks$nplots>0,]
stopifnot(nrow(blocks)>=5L)
set.seed(2026)
blocks <- blocks[sample(seq_len(nrow(blocks))),]
blocks <- blocks[order(-blocks$nplots),]
load <- rep(0L,5)
blocks$fold <- integer(nrow(blocks))
for(i in seq_len(nrow(blocks))) {
  k <- which.min(load)
  blocks$fold[i] <- k
  load[k] <- load[k]+blocks$nplots[i]
}
plots$fold <- blocks$fold[match(plots$block,blocks$block)]
cols <- c("plot","long","lat","ix","iy","block","fold")
a <- plots[order(plots$plot),cols]
b <- saved[order(saved$plot),cols]
rownames(a) <- NULL
rownames(b) <- NULL
stopifnot(identical(as.character(a$plot),as.character(b$plot)))
for(nm in c("ix","iy","block","fold")) {
  if(!identical(as.character(a[[nm]]),as.character(b[[nm]]))) {
    stop(paste("Archived fold mismatch:",nm,"count",sum(as.character(a[[nm]])!=as.character(b[[nm]]))))
  }
}
for(nm in c("long","lat")) stopifnot(isTRUE(all.equal(a[[nm]],b[[nm]],tolerance=1e-10)))
dir.create("results/v623_fold_replay",recursive=TRUE,showWarnings=FALSE)
write.csv(a,"results/v623_fold_replay/replayed_spatial_2d_block_assignments.csv",row.names=FALSE)
writeLines(c("status=PASS_EXACT_PLOT_FOLD_MAPPING",
             paste0("R_version=",R.version.string),
             paste0("plots=",nrow(a)),
             paste0("blocks=",nrow(blocks)),
             paste0("fold_counts=",paste(tabulate(a$fold,nbins=5),collapse=",")),
             "note=Replayed original coordinate-only algorithm; byte-identical CSV is not asserted"),
           "results/v623_fold_replay/summary.txt")
cat(readLines("results/v623_fold_replay/summary.txt"),sep="\n")

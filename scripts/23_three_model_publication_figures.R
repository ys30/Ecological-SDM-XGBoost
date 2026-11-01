# Reproduce publication-style three-model comparison figures from the committed CSV.
# Usage: Rscript scripts/23_three_model_publication_figures.R
d <- read.csv("results/matched_bayesian_maxent_xgboost_species.csv", check.names=FALSE)
stopifnot(nrow(d)==5L, sum(d$folds)==23L,
          all(d$folds>0), all(d$positives>0),
          all(vapply(d[c("bayesian_auc","maxent_auc","xgboost_auc",
                         "bayesian_ap","maxent_ap","xgboost_ap")],
                     function(x) all(is.finite(x)&x>=0&x<=1), logical(1))))
dir.create("results/figures", recursive=TRUE, showWarnings=FALSE)
species <- gsub(".", " ", d$species, fixed=TRUE)
models <- c("Bayesian","MaxEnt","XGBoost")
cols <- c("#4B78A8","#E0A458","#50A08B")
draw <- function(metric, title, file) {
  vals <- sapply(tolower(models), function(m) d[[paste0(m,"_",metric)]])
  stopifnot(nrow(vals)==nrow(d), ncol(vals)==3)
  svg(file, width=11.6, height=7.1, pointsize=12, bg="white")
  op <- par(mar=c(5,14,5,2), xpd=NA, family="sans")
  yy <- rev(seq_len(nrow(d)))*4
  plot(NA, xlim=c(0,1.13), ylim=c(0,max(yy)+3),
       xlab=if(metric=="auc") "ROC AUC (0–1)" else "Average precision (0–1)",
       ylab="", yaxt="n", xaxt="n", bty="n", main=title)
  axis(1, at=seq(0,1,0.2), labels=sprintf("%.1f",seq(0,1,0.2)))
  abline(v=seq(0,1,0.2), col="#E8EBEF", lty=1)
  axis(2, at=yy, labels=sprintf("%s   (folds=%d; positives=%d)",
       species,d$folds,d$positives), las=1, tick=FALSE, cex.axis=0.78)
  for(i in seq_len(nrow(d))) for(j in 1:3) {
    y <- yy[i]+(2-j)*0.77
    rect(0,y-0.29,vals[i,j],y+0.29,col=cols[j],border=NA)
    text(vals[i,j]+0.012,y,sprintf("%.3f",vals[i,j]),adj=0,cex=0.7)
  }
  legend("topright", legend=models, fill=cols, bty="n", horiz=TRUE, inset=c(0,-0.06))
  mtext("Matched held-out spatial folds; original-fit Bayesian predictions; descriptive fold means",
        side=1,line=3.7,cex=0.73,col="#555555")
  par(op); dev.off()
}
draw("auc","Matched three-model spatial CV: ROC AUC",
     "results/figures/23_three_model_auc_publication.svg")
draw("ap","Matched three-model spatial CV: Average precision",
     "results/figures/23_three_model_ap_publication.svg")

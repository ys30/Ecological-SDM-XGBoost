# Calibration and threshold sensitivity on fixed plot-grouped holdout predictions.
# Diagnostic analysis only: threshold selection on this same holdout is
# exploratory and must NOT be reported as independently validated performance.
suppressPackageStartupMessages(library(ggplot2))
d <- read.csv("results/plot_holdout_predictions.csv")
stopifnot(all(c("observed","predicted_probability","plot") %in% names(d)),
          all(d$observed %in% c(0,1)),
          all(is.finite(d$predicted_probability)),
          all(d$predicted_probability >= 0 & d$predicted_probability <= 1))
dir.create("results/figures",recursive=TRUE,showWarnings=FALSE)
y <- d$observed; p <- d$predicted_probability
brier <- mean((p-y)^2)
prevalence <- mean(y)
# Equal-count bins; repeated probabilities can share a bin boundary but
# quantile-rank bins remain deterministic and avoid empty equal-width bins.
ord <- order(p,seq_along(p))
bin <- integer(length(p))
bin[ord] <- pmin(10L,ceiling(seq_along(p)*10/length(p)))
cal <- aggregate(cbind(predicted=p,observed=y)~bin,FUN=mean)
cal$n <- as.integer(table(factor(bin,levels=1:10)))
cal$observed_se <- sqrt(cal$observed*(1-cal$observed)/cal$n)
cal$observed_lo <- pmax(0,cal$observed-1.96*cal$observed_se)
cal$observed_hi <- pmin(1,cal$observed+1.96*cal$observed_se)
write.csv(cal,"results/plot_holdout_calibration_bins.csv",row.names=FALSE)
cal_metrics <- data.frame(n=length(y),prevalence=prevalence,brier_score=brier,
  baseline_brier_score=prevalence*(1-prevalence),
  mean_predicted_probability=mean(p))
write.csv(cal_metrics,"results/plot_holdout_calibration_summary.csv",row.names=FALSE)
g <- ggplot(cal,aes(x=predicted,y=observed))+
  geom_abline(intercept=0,slope=1,linetype=2,color="grey50")+
  geom_errorbar(aes(ymin=observed_lo,ymax=observed_hi),width=.01,color="#287a69")+
  geom_point(aes(size=n),color="#287a69")+
  geom_line(color="#287a69")+
  coord_equal(xlim=c(0,1),ylim=c(0,1))+
  labs(title=sprintf("Plot-held-out reliability | Brier %.3f",brier),
       subtitle="10 equal-count probability bins; Wald intervals are descriptive only",
       x="Mean predicted presence probability",y="Observed presence fraction")+
  theme_minimal(base_size=12)
ggsave("results/figures/12_plot_holdout_calibration.png",g,width=7,height=6,dpi=160)
thresholds <- seq(0.05,0.95,by=.05)
safe <- function(a,b) if(b>0) a/b else NA_real_
tab <- do.call(rbind,lapply(thresholds,function(t){
  z <- p>=t
  tp <- sum(z & y==1);fp <- sum(z & y==0)
  fn <- sum(!z & y==1);tn <- sum(!z & y==0)
  precision <- safe(tp,tp+fp);recall <- safe(tp,tp+fn)
  data.frame(threshold=t,precision=precision,recall=recall,
    specificity=safe(tn,tn+fp),f1=safe(2*tp,2*tp+fp+fn),
    predicted_positive=sum(z),tp=tp,fp=fp,tn=tn,fn=fn)
}))
write.csv(tab,"results/plot_holdout_threshold_sensitivity.csv",row.names=FALSE)
long <- rbind(data.frame(threshold=tab$threshold,metric="Precision",value=tab$precision),
 data.frame(threshold=tab$threshold,metric="Recall",value=tab$recall),
 data.frame(threshold=tab$threshold,metric="F1",value=tab$f1))
g2 <- ggplot(long,aes(threshold,value,color=metric))+
  geom_line(linewidth=1)+geom_point(size=1.5)+
  geom_vline(xintercept=.5,linetype=2,color="grey60")+
  scale_x_continuous(breaks=seq(.1,.9,.1))+
  coord_cartesian(ylim=c(0,1))+
  labs(title="Threshold sensitivity on held-out plots",
       subtitle="Exploratory only; do not optimize threshold and evaluate on the same holdout",
       x="Decision threshold",y="Metric",color="Metric")+
  theme_minimal(base_size=12)
ggsave("results/figures/13_plot_holdout_thresholds.png",g2,width=9,height=5,dpi=160)
print(cal_metrics)
print(tab[which.min(abs(tab$threshold-.5)),])

# Reproducible five-fold geographic assignment map (no model refitting)
# Run from repository root: Rscript scripts/24_plot_spatial_fold_map.R
suppressPackageStartupMessages(library(ggplot2))
input <- "results/spatial_2d_block_assignments.csv"
stopifnot(file.exists(input))
d <- read.csv(input, stringsAsFactors = FALSE)
stopifnot(all(c("plot","long","lat","fold","block") %in% names(d)))
stopifnot(nrow(d) == 1167L, !anyNA(d[,c("plot","long","lat","fold")]))
stopifnot(!anyDuplicated(d$plot), setequal(unique(d$fold), 1:5))
stopifnot(all(is.finite(d$long)), all(is.finite(d$lat)))
stopifnot(all(d$long >= -180 & d$long <= 180), all(d$lat >= -90 & d$lat <= 90))
stopifnot(all(tapply(d$fold, d$block, function(x) length(unique(x))) == 1L))
d$fold <- factor(d$fold, levels = 1:5, labels = paste("Fold",1:5))
counts <- table(d$fold)
message(paste(names(counts), counts, sep=": ", collapse=" | "))
p <- ggplot(d, aes(long, lat, colour=fold)) +
  geom_point(size=1.45, alpha=.78) +
  coord_quickmap() +
  scale_colour_brewer(palette="Dark2", drop=FALSE) +
  labs(x="Longitude (°E)", y="Latitude (°N)", colour="Held-out fold",
       title="Geographic allocation of 1,167 survey plots",
       subtitle="Five spatial-block folds; all taxa at each plot share a fold",
       caption="Point locations and fold IDs: archived spatial_2d_block_assignments.csv.\nSpatially separated internal validation; not buffered external transfer.") +
  theme_bw(base_size=11) + theme(panel.grid.minor=element_blank(), legend.position="bottom")
dir.create("results/figures", recursive=TRUE, showWarnings=FALSE)
ggsave("results/figures/figure1_spatial_five_folds.png",p,width=8.4,height=6.5,dpi=300)
ggsave("results/figures/figure1_spatial_five_folds.pdf",p,width=8.4,height=6.5)
write.csv(data.frame(fold=names(counts),plots=as.integer(counts)),
          "results/figures/figure1_fold_counts.csv",row.names=FALSE)

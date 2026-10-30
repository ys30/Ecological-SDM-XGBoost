# Standalone portfolio EDA: run from repository root.
# Install once: install.packages(c("ggplot2", "dplyr", "tidyr"))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
d <- read.csv("data/train.csv", stringsAsFactors = FALSE)
stopifnot(all(c("Species","pres.abs","plot","tempann","rainann",
                "soildepth","topo","long","lat","easting","northing") %in% names(d)))
stopifnot(all(na.omit(d$pres.abs) %in% c(0,1)))
theme_set(theme_minimal(base_size = 13))
save_fig <- function(p, name, w = 9, h = 5) {
  ggsave(file.path("results/figures", name), p, width = w, height = h,
         dpi = 170, bg = "white")
}
# Figure 1: positive-class imbalance
p1 <- ggplot(d, aes(x = factor(pres.abs, levels = c(0,1),
                              labels = c("Absent (0)","Present (1)")))) +
  geom_bar(fill = "#356f9c", width = .65) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -.5) +
  labs(title = "Species presence is uncommon", x = NULL, y = "Observations") +
  expand_limits(y = nrow(d))
save_fig(p1, "01_class_balance.png")
# Figure 2: species counts; no claim of pairwise co-occurrence
species <- d %>% group_by(Species) %>%
  summarise(Presence_Count = sum(pres.abs == 1, na.rm = TRUE),
            Total_Count = sum(!is.na(pres.abs)), .groups = "drop") %>%
  mutate(Absence_Count = Total_Count - Presence_Count,
         Presence_Rate = Presence_Count / Total_Count)
write.csv(species, "results/species_summary_recomputed.csv", row.names = FALSE)
long_species <- species %>% select(Species, Presence_Count, Absence_Count) %>%
  pivot_longer(-Species, names_to = "Status", values_to = "Count")
p2 <- ggplot(long_species, aes(x = reorder(Species, Count, FUN = sum),
                               y = Count, fill = Status)) +
  geom_col() + coord_flip() +
  scale_fill_manual(values = c(Absence_Count = "#a8b6c6",
                               Presence_Count = "#27805c")) +
  labs(title = "Observed presence and absence by species",
       x = NULL, y = "Observations", fill = NULL)
save_fig(p2, "02_species_counts.png", 10, 6)
# Figure 3: different predictor units require free scales
env <- d %>% select(tempann, rainann, soildepth, topo) %>%
  pivot_longer(everything(), names_to = "Predictor", values_to = "Value")
p3 <- ggplot(env, aes(Value)) +
  geom_histogram(bins = 30, fill = "#6eb0cf", color = "white") +
  facet_wrap(~ Predictor, scales = "free", ncol = 2) +
  labs(title = "Environmental predictor distributions", y = "Frequency")
save_fig(p3, "03_environmental_distributions.png", 10, 7)
# Figure 4: avoid comparing coordinate scales in a single axis
coords <- d %>% select(long, lat, easting, northing) %>%
  pivot_longer(everything(), names_to = "Coordinate", values_to = "Value")
p4 <- ggplot(coords, aes(x = "", y = Value)) +
  geom_boxplot(fill = "#acd5e4", outlier.color = "#b23b3b") +
  facet_wrap(~ Coordinate, scales = "free_y", ncol = 2) +
  labs(title = "Spatial coordinate distributions", x = NULL, y = "Recorded value") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
save_fig(p4, "04_spatial_boxplots.png", 10, 7)
# Figure 5: numerical correlations; categorical codes excluded
numeric_names <- c("long","lat","rainann","soildepth","tempann",
                   "topo","easting","northing")
cm <- cor(d[, numeric_names], use = "pairwise.complete.obs")
write.csv(cm, "results/numeric_correlations.csv")
cm_long <- as.data.frame(as.table(cm))
p5 <- ggplot(cm_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() + coord_equal() +
  scale_fill_gradient2(low = "#b45a4a", mid = "white",
                       high = "#32658d", midpoint = 0, limits = c(-1,1)) +
  labs(title = "Correlations among numeric predictors",
       x = NULL, y = NULL, fill = "Pearson r") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_fig(p5, "05_predictor_correlations.png", 9, 8)
message("Saved five data-derived figures and two CSV tables in results/.")

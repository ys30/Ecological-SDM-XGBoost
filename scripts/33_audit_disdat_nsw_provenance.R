#!/usr/bin/env Rscript
# Provenance investigation: compare the repository plot table with disdat NSW
# open-forest tree evaluation data. This is a diagnostic, not an assumed match.
suppressPackageStartupMessages(library(disdat))
local <- read.csv("data/train.csv", check.names=FALSE)
stopifnot(nrow(local)==9336L, length(unique(local$plot))==1167L)
pa <- disdat::disPa("NSW", "ot")
env <- disdat::disEnv("NSW", "ot")
dir.create("results", showWarnings=FALSE)
sink("results/disdat_nsw_open_forest_provenance_check.txt")
cat("Candidate: disdat::disPa('NSW','ot') and disEnv('NSW','ot')\n")
cat("Local species-plot rows:", nrow(local), "\n")
cat("Local distinct plots:", length(unique(local$plot)), "\n")
cat("Local species:", paste(sort(unique(local$Species)),collapse=", "), "\n")
cat("Reference PA dimensions:", paste(dim(pa),collapse=" x "), "\n")
cat("Reference ENV dimensions:", paste(dim(env),collapse=" x "), "\n")
cat("PA column names:", paste(names(pa),collapse=", "), "\n")
cat("ENV column names:", paste(names(env),collapse=", "), "\n")
cat("NSW CRS from disdat:", as.character(disdat::disCRS("NSW", format="EPSG")), "\n")
cat("PA first 3 records:\n"); print(utils::head(pa,3))
cat("ENV first 3 records:\n"); print(utils::head(env,3))
cat("Local first 3 records:\n"); print(utils::head(local,3))
# Match shared environmental covariates as exact row signatures, regardless of
# reference site-ID naming. This test is strong if all plot signatures match.
shared <- intersect(c("disturb","rainann","soildepth","soilfert","tempann","topo"),names(env))
cat("Shared environmental covariates:",paste(shared,collapse=", "), "\n")
if(length(shared)>=4L) {
  unique_local <- local[!duplicated(local$plot),shared,drop=FALSE]
  sig <- function(df) do.call(paste,c(lapply(df,as.character),sep="|"))
  local_sig <- sig(unique_local)
  ref_sig <- sig(env[,shared,drop=FALSE])
  cat("Unique local plot signatures:",length(unique(local_sig)),"\n")
  cat("Unique reference signatures:",length(unique(ref_sig)),"\n")
  cat("Local plots with matching reference environment:",sum(local_sig %in% ref_sig),
      "of",nrow(unique_local),"\n")
  cat("NOTE: Environmental signature matches alone cannot prove identical species labels.\n")
} else {
  cat("Cannot compare environmental signatures: fewer than four matching fields.\n")
}
sink()
cat(readLines("results/disdat_nsw_open_forest_provenance_check.txt"),sep="\n")

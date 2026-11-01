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
cat("NSW CRS from disdat (if available):\n")
tryCatch(print(disdat::disCRS("NSW")), error=function(e) cat("CRS lookup unavailable:",conditionMessage(e),"\n"))
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

# Compare coordinate values only when the reference columns can be identified.
# Reference x/y may be projected metres rather than longitude/latitude.
loc <- local[!duplicated(local$plot), , drop=FALSE]
ref_lon <- intersect(c("long","longitude","lon"),names(env))
ref_lat <- intersect(c("lat","latitude"),names(env))
# disdat NSW x/y are geographic longitude/latitude in WGS84.
if(length(ref_lon)==0 && length(ref_lat)==0 &&
   all(c("x","y") %in% names(env))) {
  crs_txt <- tryCatch(as.character(disdat::disCRS("NSW")),error=function(e) "")
  if(grepl("longlat",crs_txt,fixed=TRUE)) {
    ref_lon <- "x"
    ref_lat <- "y"
    cat("Reference x/y interpreted as WGS84 longitude/latitude using disCRS.\\n")
  }
}
if (length(ref_lon)>0 && length(ref_lat)>0) {
  local_xy <- paste(round(loc$long,4),round(loc$lat,4),sep="|")
  ref_xy <- paste(round(env[[ref_lon[1]]],4),round(env[[ref_lat[1]]],4),sep="|")
  cat("Geographic coordinate matches:",sum(local_xy %in% ref_xy),"of",nrow(loc),"\n")
  if(length(shared)>=4L) {
    local_joint <- paste(local_xy,sig(loc[,shared,drop=FALSE]),sep="|")
    ref_joint <- paste(ref_xy,sig(env[,shared,drop=FALSE]),sep="|")
    cat("Joint coordinate-and-environment matches:",sum(local_joint %in% ref_joint),"of",nrow(loc),"\n")
  }
} else {
  cat("Geographic coordinate match not attempted: no verified lon/lat columns.\n")
  cat("Reference x/y, if present, require CRS verification before comparison.\n")
}
# Additional evidence: compare species-level occurrence totals if PA contains
# identically named binary response columns. Totals are a screening check,
# not proof that individual sites or labels are identical.
local_species <- sort(unique(as.character(local$Species)))
candidate_cols <- intersect(local_species,names(pa))
cat("Exact species-name columns in reference PA:",
    paste(candidate_cols,collapse=", "), "\n")
if(length(candidate_cols)>0L) {
  for(sp in candidate_cols) {
    vals <- suppressWarnings(as.numeric(as.character(pa[[sp]])))
    if(all(is.na(vals) | vals %in% c(0,1))) {
      cat("Species",sp,"local positives",sum(local$pres.abs[local$Species==sp]==1),
          "reference positives",sum(vals==1,na.rm=TRUE),"\n")
    } else {
      cat("Species",sp,"reference values not safely interpretable as binary; skipped\n")
    }
  }
}
cat("IMPORTANT: No claim of exact dataset identity without matched sites AND labels.\n")

# The disdat presence/absence table is WIDE (one column per species), whereas
# this repository is LONG (one row per species per plot). Report this explicitly.
cat("Reference PA is wide format; local input is long format.\n")
cat("Reference PA candidate species columns:",
    paste(setdiff(names(pa),c("group","site","x","y","long","lat")),collapse=", "), "\n")
cat("Local observed binary presences:",sum(local$pres.abs==1),"\n")
cat("Local rows per plot unique values:",paste(unique(as.integer(table(local$plot))),collapse=", "), "\n")
sink()
cat(readLines("results/disdat_nsw_open_forest_provenance_check.txt"),sep="\n")

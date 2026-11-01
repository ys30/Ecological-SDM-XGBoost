#!/usr/bin/env Rscript
# Record the package versions actually installed in a workflow run.
# This file is an environment snapshot, not proof of historical versions.
dir.create("results", recursive=TRUE, showWarnings=FALSE)
pkgs <- c("R", "xgboost", "maxnet", "brms", "pROC", "ggplot2",
          "posterior", "rstan", "Matrix", "data.table")
out <- do.call(rbind, lapply(pkgs, function(pkg) {
  if (pkg == "R") {
    return(data.frame(package="R", version=as.character(getRversion()),
                      installed=TRUE, stringsAsFactors=FALSE))
  }
  available <- requireNamespace(pkg, quietly=TRUE)
  data.frame(package=pkg,
             version=if (available) as.character(utils::packageVersion(pkg)) else NA_character_,
             installed=available, stringsAsFactors=FALSE)
}))
write.csv(out, "results/sdm_environment_versions.csv", row.names=FALSE, na="")
cat("Current runtime version inventory (not historical provenance):\n")
print(out, row.names=FALSE)

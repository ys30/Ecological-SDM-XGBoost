# Buffered Bayesian refit adapter. Reuses the archived, audited brms model.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
variant <- args[1]; fold <- as.integer(args[2]); radius <- as.numeric(args[3])
stopifnot(variant %in% c("linear","quadratic"),fold %in% 1:5,radius %in% c(5,10,20))
source_file <- "scripts/57_bayes_extension_spatial_cv.R"
s <- paste(readLines(source_file,warn=FALSE),collapse="\n")
anchor <- "stopifnot(length(intersect(unique(tr$plot),unique(te$plot)))==0)"
stopifnot(grepl(anchor,s,fixed=TRUE))
insertion <- paste0(anchor,"\n",
"tp <- te[!duplicated(te$plot),c('easting','northing')]\n",
"rp <- tr[!duplicated(tr$plot),c('plot','easting','northing')]\n",
"keep <- vapply(seq_len(nrow(rp)),function(i) {\n",
" dx <- tp$easting-rp$easting[i]; dy <- tp$northing-rp$northing[i]\n",
" all(dx*dx+dy*dy >= (",radius,"*1000)^2)\n",
"},logical(1))\n",
"tr <- tr[tr$plot %in% rp$plot[keep],,drop=FALSE]\n",
"stopifnot(nrow(tr)>0, length(unique(tr$species))==8)\n",
"cat('BUFFER',",radius,", 'FOLD',",fold,", 'TRAIN_PLOTS',length(unique(tr$plot)),'EXCLUDED',sum(!keep),'\\n')\n")
s <- sub(anchor,insertion,s,fixed=TRUE)
s <- sub('assignment_path <- if (length(args)>=3) args[3] else "results/spatial_2d_block_assignments.csv"', 'assignment_path <- "results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv"', s, fixed=TRUE)
s <- gsub("results/bayes_extensions",paste0("results/bayes_buffered/",radius,"km"),s,fixed=TRUE)
stopifnot(grepl("results/bayes_buffered",s,fixed=TRUE))
tmp <- tempfile(fileext=".R");writeLines(s,tmp)
# Original script reads commandArgs; the workflow supplies variant, fold, assignment CSV.
source(tmp,local=FALSE)

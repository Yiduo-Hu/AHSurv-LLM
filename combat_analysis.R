# Cross-cohort batch correction with ComBat.
# Reference batch is shrunk empirically; a protected covariate is preserved.

suppressPackageStartupMessages({
  library(sva)                       # ComBat (empirical Bayes shrinkage)
  library(dplyr)                     # bind_rows for stacking metadata
})

cohorts <- list(
  reference = your_reference_expr,   # numeric matrix, samples x genes
  query1    = your_query1_expr,      # all cohorts share the same gene order
  query2    = your_query2_expr
)

cohort_meta <- list(
  reference = your_reference_meta,   # data frame with `sample` + covariate cols;
  query1    = your_query1_meta,      # rows must match rownames of cohorts[[i]]
  query2    = your_query2_meta
)

ref_batch <- "reference"             # cohort used as EB reference
age_col   <- "age"                   # covariate protected by mod

stopifnot(length(unique(lapply(cohorts, colnames))) == 1L)  # gene panels aligned

combined_expr <- do.call(rbind, cohorts)
combined_meta <- bind_rows(cohort_meta)                       # same row order as above

batch <- factor(
  rep(names(cohorts), times = vapply(cohorts, nrow, integer(1))),
  levels = names(cohorts)
)

mod <- model.matrix(reformulate(age_col, intercept = FALSE),
                    data = combined_meta)                     # preserves age

combat_out <- ComBat(
  dat         = t(combined_expr),     # genes x samples
  batch       = batch,
  mod         = mod,                  # absorbs biology, isolates batch
  ref.batch   = ref_batch,            # empirical-Bayes reference cohort
  par.prior   = TRUE,
  mean.only   = FALSE,
  prior.plots = FALSE
)

corrected <- t(combat_out)            # samples x genes

offsets <- c(0L, cumsum(vapply(cohorts, nrow, integer(1))))
corrected_cohorts <- lapply(seq_along(cohorts), function(i) {
  rows <- (offsets[i] + 1L):offsets[i + 1L]
  corrected[rows, , drop = FALSE]
})
names(corrected_cohorts) <- names(cohorts)
# Differential expression analysis across age subgroups.
# For each subgroup (full / <65 / >=65), run limma-trend against Cancer vs.
# Normal, then merge the three DEG tables (keep max |log2FC| per gene,
# alphabetical output). Genes should already be restricted to the cross-
# cohort intersection (handled upstream).

suppressPackageStartupMessages({
  library(limma)        # lmFit / eBayes / topTable
  library(dplyr)        # data wrangling and merge
})

# Thresholds
PADJ_CUT           <- 0.05
FC_CUT             <- 1.0
MIN_EXPR_THRESHOLD <- 1.0
MIN_SAMPLE_FRAC    <- 0.10

# Inputs (user-supplied): genes (rows) x samples (cols) matrix and a metadata
# data frame with `sample`, `group`, and `age` columns.
expr <- your_expr_matrix
meta <- your_metadata_frame

meta <- meta[!is.na(meta$age), ]
expr <- expr[, meta$sample, drop = FALSE]

splits <- list(
  all    = meta$sample,
  lt65   = meta$sample[meta$age <  65],
  gte65  = meta$sample[meta$age >= 65]
)

# Single-group DEA: low-expression filter -> cancer/normal factor ->
# limma-trend -> padj + |log2FC| filter.
dea_one <- function(samples) {
  keep    <- rowSums(expr[, samples, drop = FALSE] >= MIN_EXPR_THRESHOLD) >=
             ceiling(MIN_SAMPLE_FRAC * length(samples))
  sub_mat <- expr[keep, samples, drop = FALSE]

  sub_meta <- data.frame(
    sample = samples,
    group  = factor(meta$group[match(samples, meta$sample)],
                    levels = c("normal", "cancer")),
    stringsAsFactors = FALSE
  )

  fit  <- lmFit(sub_mat, model.matrix(~ group, data = sub_meta))
  fit2 <- eBayes(fit, trend = TRUE)
  tt   <- topTable(fit2, coef = 2, number = Inf, sort.by = "none")

  sig <- tt %>% filter(!is.na(adj.P.Val) & adj.P.Val < PADJ_CUT & abs(logFC) > FC_CUT)
  data.frame(
    gene           = rownames(sig),
    log2FoldChange = sig$logFC,
    padj           = sig$adj.P.Val,
    stringsAsFactors = FALSE
  )
}

sig_list <- lapply(splits, dea_one)

# Merge across subgroups: bind_rows + group_by(gene) keeps max |log2FC|;
# arrange(gene) sorts alphabetically.
union_info <- bind_rows(sig_list, .id = "source") %>%
  mutate(abs_l2fc = abs(log2FoldChange)) %>%
  group_by(gene) %>%
  slice_max(abs_l2fc, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(gene) %>%
  select(gene, log2FoldChange, padj, source)

# Output: union_info$gene is the alphabetized DEG list; union_info is the
# full table with log2FoldChange, padj, and source subgroup.
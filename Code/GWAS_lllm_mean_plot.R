library(data.table)
library(Cairo)
library(qqman)

# --- Set working directory ---
setwd('/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_llm_chr/')

# --- Helper: map chr labels to integers ---
map_chr_to_int <- function(x) {
  x2 <- tolower(gsub("^chr", "", as.character(x)))
  x2[x2 %in% c("x")]  <- "23"
  x2[x2 %in% c("y")]  <- "24"
  x2[x2 %in% c("mt","m")] <- "25"
  suppressWarnings(as.integer(x2))
}

# --- Load all chromosome results ---
combined <- rbindlist(lapply(1:22, function(chr) {
  file <- paste0("chr", chr, "_llmmean_ord_assoc.txt")
  if (file.exists(file)) {
    dat <- fread(file)
    dat <- dat[!is.na(Pvalue) & Pvalue > 0]
    return(dat)
  } else NULL
}), use.names = TRUE, fill = TRUE)

stopifnot(nrow(combined) > 0)

# --- Work on a copy ---
gwas <- copy(combined)

# --- Ensure core columns: SNP, CHR, BP (parse from Info if needed) ---
# SNP (prefer SNP, else Marker, else synthesize)
if (!"SNP" %in% names(gwas) && "Marker" %in% names(gwas)) {
  gwas[, SNP := as.character(Marker)]
} else if (!"SNP" %in% names(gwas)) {
  gwas[, SNP := paste0("var_", .I)]
} else {
  gwas[, SNP := as.character(SNP)]
}

# CHR/BP from Info if missing (Info like "CHR:BP:REF:ALT")
need_chr <- !"CHR" %in% names(gwas)
need_bp  <- !"BP"  %in% names(gwas)
if ((need_chr || need_bp) && "Info" %in% names(gwas)) {
  parts <- tstrsplit(gwas$Info, ":", fixed = TRUE, fill = NA)
  if (need_chr && length(parts) >= 1) gwas[, CHR := map_chr_to_int(parts[[1]])]
  if (need_bp  && length(parts) >= 2) gwas[, BP  := as.integer(parts[[2]])]
  # Optional: fill REF/ALT if absent
  if (!"REF" %in% names(gwas) && length(parts) >= 3) gwas[, REF := parts[[3]]]
  if (!"ALT" %in% names(gwas) && length(parts) >= 4) gwas[, ALT := parts[[4]]]
}

# Fallbacks for common alt names
if (!"CHR" %in% names(gwas)) {
  alt_chr <- intersect(c("#CHROM","CHROM","Chrom","chrom"), names(gwas))
  if (length(alt_chr)) gwas[, CHR := map_chr_to_int(get(alt_chr[1]))]
}
if (!"BP" %in% names(gwas)) {
  alt_bp <- intersect(c("POS","Position","pos","BP37","BP38"), names(gwas))
  if (length(alt_bp)) gwas[, BP := as.integer(get(alt_bp[1]))]
}

# Enforce types & filter unusable rows
gwas[, CHR := as.integer(CHR)]
gwas[, BP  := as.integer(BP)]
gwas <- gwas[!is.na(Pvalue) & Pvalue > 0]
gwas <- gwas[complete.cases(CHR, BP)]
gwas <- gwas[CHR %in% 1:25]

# === Lambda GC ===
compute_lambda <- function(p_values) {
  chisq_stats <- qchisq(1 - p_values, df = 1)
  median(chisq_stats, na.rm = TRUE) / qchisq(0.5, df = 1)
}
lambda_gc <- compute_lambda(gwas$Pvalue)
cat("LLM mean — Lambda GC:", lambda_gc, "\n")

# === Lambda-adjusted p ===
adjust_pvalues <- function(dt, lambda_gc) {
  chisq_obs <- qchisq(dt$Pvalue, df = 1, lower.tail = FALSE)
  chisq_adj <- chisq_obs / lambda_gc
  # numerically stable via log.p, then exp
  log_pval  <- pchisq(chisq_adj, df = 1, lower.tail = FALSE, log.p = TRUE)
  dt[, P_lambda_adjusted := exp(log_pval)]
  dt
}
gwas <- adjust_pvalues(gwas, lambda_gc)

# --- QQ plot: unadjusted ---
CairoPNG("llmmean_qqplot_unadjusted.png", width = 800, height = 800)
qq(gwas$Pvalue, main = "QQ Plot (Unadjusted) - LLM mean GWAS")
dev.off()

# --- QQ plot: adjusted ---
CairoPNG("llmmean_qqplot_lambdaGC.png", width = 800, height = 800)
qq(gwas$P_lambda_adjusted, main = paste0("QQ Plot (λGC Adjusted, λ = ", round(lambda_gc, 3), ") - LLM mean"))
dev.off()

# --- (Optional) Manhattan plots ---
# CairoPNG("llmmean_manhattan_unadjusted.png", width = 1200, height = 600)
# manhattan(
#   as.data.frame(gwas[, .(CHR, BP, SNP, P = Pvalue)]),
#   col = c("dodgerblue", "firebrick"),
#   genomewideline = -log10(5e-8),
#   suggestiveline = -log10(1e-5),
#   main = "Manhattan Plot (Unadjusted) - LLM mean"
# )
# dev.off()
#
# CairoPNG("llmmean_manhattan_lambdaGC.png", width = 1200, height = 600)
# manhattan(
#   as.data.frame(gwas[, .(CHR, BP, SNP, P = P_lambda_adjusted)]),
#   col = c("dodgerblue", "firebrick"),
#   genomewideline = -log10(5e-8),
#   suggestiveline = -log10(1e-5),
#   main = paste0("Manhattan Plot (λGC Adjusted, λ = ", round(lambda_gc, 3), ") - LLM mean")
# )
# dev.off()

# --- Save: full table with adjusted p-values ---
fwrite(gwas, "combined_GRABgwas_llmmean_lambda_adjusted.tsv", sep = "\t")

# --- Save: significant hits ---
fwrite(gwas[Pvalue < 5e-8], "llmmean_significant_snps_unadjusted.tsv", sep = "\t")
fwrite(gwas[P_lambda_adjusted < 5e-8], "llmmean_significant_snps_lambdaGC.tsv", sep = "\t")

# --- Save: LocusZoom-ready tables (rsID, CHR, BP, P) ---
lz_unadj <- gwas[, .(rsID = SNP, CHR, BP, P = Pvalue)][order(CHR, BP)]
lz_adj   <- gwas[, .(rsID = SNP, CHR, BP, P = P_lambda_adjusted)][order(CHR, BP)]
fwrite(lz_unadj, "llmmean_locuszoom_unadjusted.txt", sep = "\t", quote = FALSE, na = "NA")
fwrite(lz_adj,   "llmmean_locuszoom_lambda_adjusted.txt", sep = "\t", quote = FALSE, na = "NA")


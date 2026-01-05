library(data.table)
library(Cairo)
library(qqman)

# --- Set working directory ---
setwd('/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_cci_chr/')

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
  file <- paste0("chr", chr, "_cci_ord_assoc.txt")
  if (file.exists(file)) {
    dat <- fread(file)
    return(dat)
  } else {
    NULL
  }
}), use.names = TRUE, fill = TRUE)

stopifnot(nrow(combined) > 0)

# --- Basic cleaning & keep valid p-values ---
gwas <- combined[!is.na(Pvalue) & Pvalue > 0]

# --- Ensure core columns exist: SNP, CHR, BP ---
# SNP (prefer 'SNP', else from 'Marker')
if (!"SNP" %in% names(gwas) && "Marker" %in% names(gwas)) {
  gwas[, SNP := as.character(Marker)]
} else if (!"SNP" %in% names(gwas)) {
  # If neither exists, synthesize from row number
  gwas[, SNP := paste0("var_", .I)]
} else {
  gwas[, SNP := as.character(SNP)]
}

# CHR/BP: if missing, try parse from Info ("CHR:BP:REF:ALT")
need_chr <- !"CHR" %in% names(gwas)
need_bp  <- !"BP"  %in% names(gwas)
if ((need_chr || need_bp) && "Info" %in% names(gwas)) {
  parts <- tstrsplit(gwas$Info, ":", fixed = TRUE, fill = NA)
  if (need_chr && length(parts) >= 1) gwas[, CHR := map_chr_to_int(parts[[1]])]
  if (need_bp  && length(parts) >= 2) gwas[, BP  := as.integer(parts[[2]])]
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

# Enforce types & filter unusable
gwas[, CHR := as.integer(CHR)]
gwas[, BP  := as.integer(BP)]
gwas <- gwas[complete.cases(CHR, BP)]
gwas <- gwas[CHR %in% 1:25]

# === Define lambda GC calculator ===
compute_lambda <- function(p_values) {
  chisq_stats <- qchisq(1 - p_values, df = 1)
  median(chisq_stats, na.rm = TRUE) / qchisq(0.5, df = 1)
}

# === Compute lambda ===
lambda_gc <- compute_lambda(gwas$Pvalue)
cat("Lambda GC:", lambda_gc, "\n")

# === Define lambda-adjusted p-value function ===
adjust_pvalues <- function(dt, lambda_gc) {
  chisq_obs <- qchisq(dt$Pvalue, df = 1, lower.tail = FALSE)
  chisq_adj <- chisq_obs / lambda_gc
  # (use log.p for stability, then exp)
  log_pval  <- pchisq(chisq_adj, df = 1, lower.tail = FALSE, log.p = TRUE)
  dt[, P_lambda_adjusted := exp(log_pval)]
  dt
}

# === Apply lambda-adjusted p-values ===
gwas <- adjust_pvalues(gwas, lambda_gc)

# --- QQ plot: unadjusted ---
CairoPNG("qqplot_unadjusted.png", width = 800, height = 800)
qq(gwas$Pvalue, main = "QQ Plot (Unadjusted) - CCI GWAS")
dev.off()

# --- QQ plot: adjusted ---
CairoPNG("qqplot_adjusted_lambdaGC.png", width = 800, height = 800)
qq(gwas$P_lambda_adjusted, main = paste0("QQ Plot (λGC Adjusted, λ = ", round(lambda_gc, 3), ")"))
dev.off()

# --- (Optional) Manhattan plots ---
# CairoPNG("manhattan_unadjusted.png", width = 1200, height = 600)
# manhattan(
#   as.data.frame(gwas[, .(CHR, BP, SNP, P = Pvalue)]),
#   col = c("dodgerblue", "firebrick"),
#   genomewideline = -log10(5e-8),
#   suggestiveline = -log10(1e-5),
#   main = "Manhattan Plot (Unadjusted)"
# )
# dev.off()
#
# CairoPNG("manhattan_lambda_adjusted.png", width = 1200, height = 600)
# manhattan(
#   as.data.frame(gwas[, .(CHR, BP, SNP, P = P_lambda_adjusted)]),
#   col = c("dodgerblue", "firebrick"),
#   genomewideline = -log10(5e-8),
#   suggestiveline = -log10(1e-5),
#   main = paste0("Manhattan Plot (λGC Adjusted, λ = ", round(lambda_gc, 3), ")")
# )
# dev.off()

# --- Save: full combined table with adjusted p-values ---
fwrite(gwas, "combined_GRABgwas_CCI_lambda_adjusted.tsv", sep = "\t")

# --- Save: significant hits (unadjusted & adjusted) ---
fwrite(gwas[Pvalue < 5e-8], "significant_snps_unadjusted.tsv", sep = "\t")
fwrite(gwas[P_lambda_adjusted < 5e-8], "significant_snps_lambdaGC_adjusted.tsv", sep = "\t")

# --- Save: LocusZoom-ready tables (rsID, CHR, BP, P) ---
lz_unadj <- gwas[, .(rsID = SNP, CHR, BP, P = Pvalue)][order(CHR, BP)]
lz_adj   <- gwas[, .(rsID = SNP, CHR, BP, P = P_lambda_adjusted)][order(CHR, BP)]
fwrite(lz_unadj, "cci_locuszoom_unadjusted.txt", sep = "\t", quote = FALSE, na = "NA")
fwrite(lz_adj,   "cci_locuszoom_lambda_adjusted.txt", sep = "\t", quote = FALSE, na = "NA")

# --- Save: small summary of lambda & counts ---
summary_dt <- data.table(
  lambda_gc = lambda_gc,
  n_total   = nrow(gwas),
  n_sig_unadjusted = nrow(gwas[Pvalue < 5e-8]),
  n_sig_adjusted   = nrow(gwas[P_lambda_adjusted < 5e-8])
)
fwrite(summary_dt, "summary_lambda_counts.tsv", sep = "\t")

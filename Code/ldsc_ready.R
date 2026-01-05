# quick & dirty: build LDSC-ready for BOTH CCI and LLM-mean using your files
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(stringr)
})

# =========================
# CONFIG — edit paths here
# =========================
freq_path      <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/combined_freq_EUR.txt"

# locuszoom (already made)
lz_cci_path    <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_cci_chr/cci_locuszoom_unadjusted.txt"
lz_llm_path    <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_llm_chr/llmmean_locuszoom_unadjusted.txt"

# full GWAS (with beta/se/P, and Info for REF/ALT)
gwas_cci_path  <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_cci_chr/combined_GRABgwas_CCI_lambda_adjusted.tsv"
gwas_llm_path  <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR/gwas_llm_chr/combined_GRABgwas_llmmean_lambda_adjusted.tsv"

# outputs land here
out_dir        <- "/mnt/vstor/SOM_EPBI_XXZ10/raghav/LLM_Comorbidity/Genotype_data/gwas_results/GRAB_GWAS/synth_data/EUR"

# switches
USE_ADJUSTED_P <- FALSE   # TRUE -> prefer P_lambda_adjusted if present
FILTER_TO_LZ   <- FALSE   # TRUE -> keep only SNPs present in LZ file rsID column


# ==============
# tiny helpers
# ==============
pick_col <- function(dt, choices, required = TRUE) {
  hit <- intersect(choices, names(dt))
  if (length(hit) == 0) {
    if (required) stop(sprintf("Missing: %s\nHave: %s", paste(choices, collapse=", "), paste(names(dt), collapse=", ")))
    return(NULL)
  }
  hit[1]
}

# Info like "CHR:BP:REF:ALT"
parse_info_ref_alt <- function(info_vec) {
  parts <- tstrsplit(info_vec, ":", fixed = TRUE, fill = NA)
  data.table(
    REF = toupper(as.character(parts[[3]])),
    ALT = toupper(as.character(parts[[4]]))
  )
}

make_ldsc_ready <- function(gwas_dt, freq_dt = NULL, prefer_adjusted_p = FALSE, restrict_ids = NULL) {
  # columns
  snp_col <- pick_col(gwas_dt, c("SNP","rsID","rsid","variant_id"))
  chr_col <- if ("CHR" %in% names(gwas_dt)) "CHR" else pick_col(gwas_dt, c("#CHROM","CHROM","Chrom","chrom"))
  bp_col  <- pick_col(gwas_dt, c("BP","POS","Position","pos","BP37","BP38"))
  beta_col<- pick_col(gwas_dt, c("beta","BETA","Effect","EffectSize"))
  se_col  <- pick_col(gwas_dt, c("seBeta","SE","StdErr","se"))
  p_col   <- if (prefer_adjusted_p && "P_lambda_adjusted" %in% names(gwas_dt)) "P_lambda_adjusted" else pick_col(gwas_dt, c("Pvalue","P","pval","PVAL","p"))
  
  setnames(gwas_dt, snp_col, "SNP")
  if (chr_col != "CHR") setnames(gwas_dt, chr_col, "CHR")
  if (bp_col  != "BP")  setnames(gwas_dt, bp_col,  "BP")
  
  gwas_dt <- unique(gwas_dt, by = "SNP")
  gwas_dt[, CHR := as.integer(CHR)]
  gwas_dt[, BP  := as.integer(BP)]
  
  # A1/A2 (prefer provided; else derive from Info)
  if (!any(c("A1","EA","ALT","Effect_Allele") %in% names(gwas_dt)) ||
      !any(c("A2","NEA","REF","Other_Allele") %in% names(gwas_dt))) {
    if (!"Info" %in% names(gwas_dt)) stop("No A1/A2 and no Info to derive REF/ALT")
    alle <- parse_info_ref_alt(gwas_dt$Info)
    gwas_dt[, A1 := alle$ALT]  # effect allele = ALT
    gwas_dt[, A2 := alle$REF]
  } else {
    a1_col <- pick_col(gwas_dt, c("A1","EA","ALT","Effect_Allele"))
    a2_col <- pick_col(gwas_dt, c("A2","NEA","REF","Other_Allele"))
    gwas_dt[, A1 := toupper(get(a1_col))]
    gwas_dt[, A2 := toupper(get(a2_col))]
  }
  
  base <- gwas_dt[, .(
    CHR, BP, SNP, A1, A2,
    BETA = as.numeric(get(beta_col)),
    SE   = as.numeric(get(se_col)),
    P    = as.numeric(get(p_col))
  )][P > 0]
  base <- base[complete.cases(CHR, BP, SNP, A1, A2, BETA, SE, P)]
  
  if (!is.null(restrict_ids)) base <- base[SNP %in% restrict_ids]
  
  # frequencies (coalesce external freq then AltFreq)
  base[, FREQ := NA_real_]
  if (!is.null(freq_dt) && nrow(freq_dt)) {
    snp_freq_col <- pick_col(freq_dt, c("SNP","rsid","rsID","variant_id"))
    setnames(freq_dt, snp_freq_col, "SNP")
    maf_col <- pick_col(freq_dt, c("MAF","AF","freq","FRQ","A1_FREQUENCY","A1_FREQ","ALT_AF"), required = FALSE)
    if (!is.null(maf_col)) {
      freq_sub <- unique(freq_dt[, .(SNP, FREQ_ext = as.numeric(get(maf_col)))])
      base <- merge(base, freq_sub, by = "SNP", all.x = TRUE, sort = FALSE)
      base[!is.na(FREQ_ext), FREQ := FREQ_ext]
      base[, FREQ_ext := NULL]
    }
  }
  if ("AltFreq" %in% names(gwas_dt)) {
    altfreq_map <- unique(gwas_dt[, .(SNP, FREQ_alt = as.numeric(AltFreq))])
    base <- merge(base, altfreq_map, by = "SNP", all.x = TRUE, sort = FALSE)
    base[is.na(FREQ) & !is.na(FREQ_alt), FREQ := FREQ_alt]
    base[, FREQ_alt := NULL]
  }
  
  base <- base[complete.cases(FREQ)]
  setcolorder(base, c("CHR","BP","SNP","A1","A2","FREQ","BETA","SE","P"))
  base[]
}

write_ldsc_pair <- function(ldsc_dt, prefix, out_dir) {
  out_txt <- file.path(out_dir, paste0(prefix, "_ldsc_ready.txt"))
  out_gz  <- paste0(out_txt, ".gz")
  fwrite(ldsc_dt, out_txt, sep = "\t", quote = FALSE)
  gz_con <- gzfile(out_gz, "w")
  write.table(ldsc_dt, file = gz_con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz_con)
  message("wrote: ", out_txt, " and .gz")
}

# freq (optional)
freq <- tryCatch({
  if (!is.null(freq_path) && nzchar(freq_path)) fread(freq_path) else NULL
}, error = function(e) NULL)

# ---- CCI ----
lz_cci   <- fread(lz_cci_path)         # rsID, CHR, BP, P  (unused unless FILTER_TO_LZ)
gwas_cci <- fread(gwas_cci_path)
cci_ids  <- if (FILTER_TO_LZ) unique(lz_cci$rsID) else NULL
cci_ldsc <- make_ldsc_ready(gwas_cci, freq_dt = freq, prefer_adjusted_p = USE_ADJUSTED_P, restrict_ids = cci_ids)
write_ldsc_pair(cci_ldsc, "EUR_cci", out_dir)

# ---- LLM-mean ----
lz_llm   <- fread(lz_llm_path)
gwas_llm <- fread(gwas_llm_path)
llm_ids  <- if (FILTER_TO_LZ) unique(lz_llm$rsID) else NULL
llm_ldsc <- make_ldsc_ready(gwas_llm, freq_dt = freq, prefer_adjusted_p = USE_ADJUSTED_P, restrict_ids = llm_ids)
write_ldsc_pair(llm_ldsc, "EUR_llmmean", out_dir)

message("done.")

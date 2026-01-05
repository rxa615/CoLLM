
# ================================
# Setup
# ================================
library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(lubridate)
library(readr)
library(tidyr)

# --- Set working directory (edit if needed) ---
setwd('/Users/raghavawasthi/Desktop/ProtienBioMarker/comorbidity_full_ukbb/')

# ================================
# Load & Merge
# ================================
df_raw <- read.csv('gpt_gemini_deepseek_cci_eci_comorbidity_scores.csv')
llm_mean_dis <- read.csv('df_ann_distilled_llm_mean_score.csv')  
gpt_dis <- read.csv('df_ann_distilled_score_gpt4o.csv')
gemini_dis <- read.csv('df_ann_distilled_score_gemini.csv')
deepseek_dis <- read.csv('df_ann_distilled_score_deepseek.csv')


# Merge distilled predictions into main frame
df_raw <- Reduce(function(x, y) merge(x, y, by = "participant_id", all.x = TRUE),
                 list(df_raw, gpt_dis, gemini_dis, deepseek_dis, llm_mean_dis))
remove_pid <- read.csv('w81097_20250818.csv',header = F)
df_raw <- df_raw[-which(df_raw$participant_id %in% remove_pid$V1),]

# Select all distilled LLM scores
distilled_scores <- c("score_gpt4o_distilled",
                      "score_gemini_distilled",
                      "score_deepseek_distilled",
                      "llm_mean_score_distilled")

# Function to compute correlation and tidy output
get_corr <- function(x, y, data) {
  res <- cor.test(data[[x]], data[[y]], method = "pearson")
  data.frame(
    Predictor = x,
    Target    = y,
    Correlation = res$estimate,
    CI_lower    = res$conf.int[1],
    CI_upper    = res$conf.int[2],
    p_value     = res$p.value
  )
}

# Run correlations for each distilled score vs Charlson and Elixhauser
results <- do.call(rbind, lapply(distilled_scores, function(sc) {
  rbind(
    get_corr(sc, "charlson_score", df_raw),
    get_corr(sc, "elixhauser_score", df_raw)
  )
}))

print(results, row.names = FALSE)



write.table(df_raw$participant_id,
            file = "keep_participants.txt",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)

df_raw_1 = df_raw[,c("sex","ref_age","llm_mean_score_distilled")]
#write.csv(df_raw_1,'/Users/raghavawasthi/Desktop/webapp/ukbscores.csv',row.names = F)

# write.csv(df_raw,'llm_distil_gpt_deepseek_gemini_allmean_cci_eci_comorbidity_scores.csv',row.names = F)
# ================================
# Survival fields
# ================================
df_raw <- df_raw %>%
  mutate(
    date_of_death       = lubridate::ymd(date_of_death),
    last_diagnosis_date = lubridate::ymd(last_diagnosis_date),
    event_orig          = ifelse(!is.na(date_of_death), 1, 0),
    age_group = case_when(
      ref_age < 50 ~ "<50",
      ref_age >= 50 & ref_age < 60 ~ "50-60",
      ref_age >= 60 & ref_age < 70 ~ "60-70",
      ref_age >= 70 & ref_age < 80 ~ "70-80",
      ref_age >= 80 ~ "80+"
    ),
    full_time = as.numeric(ifelse(
      event_orig == 1,
      date_of_death - last_diagnosis_date,
      lubridate::ymd("2022-11-30") - last_diagnosis_date
    )),
    surv_obj = survival::Surv(full_time, event_orig)
  ) %>%
  dplyr::filter(full_time >= 0)

# ================================
# Predictors (only 5)
# ================================
predictors <- c(
  "charlson_score",
  "elixhauser_score",
  "score_gpt4o_distilled",
  "score_gemini_distilled",
  "score_deepseek_distilled",
  "llm_mean_score_distilled"
)

predictors <- predictors[predictors %in% names(df_raw)]
if (length(predictors) == 0) stop("No predictor columns found.")

age_groups <- unique(df_raw$age_group)

# ================================
# Cox by age group (with guards)
# ================================
results_agegroup <- list()

for (age in age_groups) {
  df_sub <- df_raw %>% dplyr::filter(age_group == age)
  if (nrow(df_sub) < 20) next  # too small to fit reliably
  
  for (pred in predictors) {
    covars <- c("sex", "ref_age", "ethnicity")
    covars <- covars[covars %in% names(df_sub)]     # drop missing covars
    rhs <- paste(c(pred, covars), collapse = " + ")
    fml <- as.formula(paste("surv_obj ~", rhs))
    
    fit <- try(coxph(fml, data = df_sub), silent = TRUE)
    if (inherits(fit, "try-error")) next
    
    sm <- summary(fit)
    C   <- as.numeric(sm$concordance["C"])
    seC <- as.numeric(sm$concordance["se(C)"])
    lower <- C - 1.96 * seC
    upper <- C + 1.96 * seC
    
    results_agegroup[[length(results_agegroup) + 1]] <- data.frame(
      Age_Group   = age,
      Predictor   = pred,
      Concordance = round(C, 3),
      C_lower     = round(lower, 3),
      C_upper     = round(upper, 3)
    )
  }
}

agegroup_summary <- dplyr::bind_rows(results_agegroup) %>%
  dplyr::mutate(
    Predictor = dplyr::case_when(
      Predictor == "charlson_score"            ~ "Charlson",
      Predictor == "elixhauser_score"          ~ "Elixhauser",
      Predictor == "score_gpt4o_distilled"     ~ "GPT-4o (distilled)",
      Predictor == "score_gemini_distilled"    ~ "Gemini (distilled)",
      Predictor == "score_deepseek_distilled"  ~ "DeepSeek (distilled)",
      Predictor == "llm_mean_score_distilled"  ~ "Mean of 3 LLMs (distilled)",
      TRUE ~ Predictor
    )
  )


# order age groups
agegroup_summary$Age_Group <- factor(
  agegroup_summary$Age_Group,
  levels = c("<50", "50-60", "60-70", "70-80", "80+"),
  ordered = TRUE
)



# ----------------
# Colors (6 preds)
# ----------------
pred_colors <- c(
  "Charlson"                   = "#E69F00",  # warm amber
  "Elixhauser"                 = "#56B4E9",
  "GPT-4o (distilled)"         = "#009E73",
  "Gemini (distilled)"         = "#CC79A7",
  "DeepSeek (distilled)"       = "#D55E00",
  "Mean of 3 LLMs (distilled)" = "#000000"   # black line
)

# (optional) enforce legend order to match colors
agegroup_summary$Predictor <- factor(
  agegroup_summary$Predictor,
  levels = names(pred_colors)
)

# -------------
# Plot (updated)
# -------------

p1 <- ggplot(
  agegroup_summary,
  aes(
    x = Age_Group, y = Concordance,
    color = Predictor, fill = Predictor, group = Predictor
  )
) +
  
  # Smooth clean lines + large points
  geom_line(linewidth = 1.4) +
  geom_point(size = 3.5, stroke = 0.7) +
  
  # Clean BioRender-style confidence ribbon
  geom_ribbon(
    aes(ymin = C_lower, ymax = C_upper),
    alpha = 0.12,  # softer shading
    color = NA
  ) +
  
  # Labels
  labs(
    x = "Age Group",
    y = "Concordance Index (C-index)",
    color = "Model",
    fill = "Model"
  ) +
  
  # Y-axis limits
  scale_y_continuous(
    limits = c(0.75, 0.92),
    breaks = seq(0.75, 0.92, 0.05)
  ) +
  
  # Colors
  scale_color_manual(values = pred_colors, drop = FALSE) +
  scale_fill_manual(values = pred_colors, drop = FALSE) +
  
  # Nature–Medicine / BioRender-style theme
  theme_classic(base_size = 16) +
  
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 15),
    legend.text  = element_text(size = 13),
    
    axis.title.x = element_text(face = "bold", size = 16, margin = margin(t = 10)),
    axis.title.y = element_text(face = "bold", size = 16, margin = margin(r = 10)),
    
    axis.text.x  = element_text(face = "bold", size = 13),
    axis.text.y  = element_text(face = "bold", size = 13),
    
    plot.margin  = margin(15, 20, 15, 15),
    panel.spacing = unit(1, "lines")
  )

print(p1)
ggsave(
  filename = "new_survival_plot.png",
  plot = p1,
  dpi = 400,
  width = 9,
  height = 6,
  units = "in"
)

## ================================
# Histograms (all six scores)
# ================================
library(dplyr)
library(tidyr)
library(ggplot2)

score_cols <- c(
  "charlson_score",
  "elixhauser_score",
  "score_gpt4o_distilled",
  "score_gemini_distilled",
  "score_deepseek_distilled",
  "llm_mean_score_distilled"
)

# keep only available columns
score_cols <- intersect(score_cols, names(df_raw))
if (length(score_cols) == 0) stop("No score columns found in df_raw.")

df_long <- df_raw %>%
  dplyr::select(dplyr::all_of(score_cols)) %>%
  dplyr::rename(
    Charlson                   = charlson_score,
    Elixhauser                 = elixhauser_score,
    `GPT-4o (distilled)`       = score_gpt4o_distilled,
    `Gemini (distilled)`       = score_gemini_distilled,
    `DeepSeek (distilled)`     = score_deepseek_distilled,
    `Mean of 3 LLMs (distilled)` = llm_mean_score_distilled
  ) %>%
  tidyr::pivot_longer(
    cols = dplyr::everything(),
    names_to = "ScoreType",
    values_to = "Score"
  ) %>%
  dplyr::filter(!is.na(Score))

# use same palette as the C-index plot (subset to what's present)
score_colors <- c(
  "Charlson"                   = "#d95f02",
  "Elixhauser"                 = "#7570b3",
  "GPT-4o (distilled)"         = "#1b9e77",
  "Gemini (distilled)"         = "#66a61e",
  "DeepSeek (distilled)"       = "#e7298a",
  "Mean of 3 LLMs (distilled)" = "#377eb8"
)
score_colors <- score_colors[names(score_colors) %in% unique(df_long$ScoreType)]

p_hist <- ggplot(df_long, aes(x = Score, fill = ScoreType)) +
  geom_histogram(bins = 20, color = "black", alpha = 0.7) +
  facet_wrap(~ ScoreType, scales = "free", ncol = 3) +
  scale_fill_manual(values = score_colors, guide = "none") +
  theme_minimal(base_size = 14) +
  labs(title = "Distribution of Risk Scores", x = "Score", y = "Count") +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text  = element_text(face = "bold")
  )

print(p_hist)

# ---------------------------------------------------------------------------
# static_mimic_to_icht_aumc_d0-7.R – cumulative-day static (all-values) models
# Primary:   MIMIC model -> ICHT + AUMC (no recal)
# Secondary: Recalibration per site (train site -> test same-site holdout)
# Trajectories: ICHT + AUMC daily means (±SE)
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr)
  library(xgboost); library(caret); library(pROC); library(ggplot2); library(readr)
})

# ---------------- Output structure -----------------------------------------
root_dir <- "Class prediction validation v4"
dir.create(root_dir, showWarnings = FALSE)

dir_primary_mc   <- file.path(root_dir, "Primary", "multiclass")
dir_primary_bin  <- file.path(root_dir, "Primary", "one_vs_other")
dir_recal_mc     <- file.path(root_dir, "Recalibration", "multiclass")
dir_recal_bin    <- file.path(root_dir, "Recalibration", "one_vs_other")
dir_traj_icht    <- file.path(root_dir, "Trajectories", "ICHT")
dir_traj_aumc    <- file.path(root_dir, "Trajectories", "AUMC")
dir_traj <- file.path(root_dir, "Trajectories")
for (p in c(dir_primary_mc, dir_primary_bin, dir_recal_mc, dir_recal_bin,
            dir_traj_icht, dir_traj_aumc)) dir.create(p, recursive = TRUE, showWarnings = FALSE)

# ---------------- Model export directory (for sharing) ----------------------
# Models saved as JSON (xgboost >= 1.6); loadable with xgb.load()
script_dir <- normalizePath(file.path(
  "D:/Dropbox/PhD/Public Git/AHRF-Trajectories-code-/Second submission code/models"
), mustWork = FALSE)
dir_mod_frozen_mc  <- file.path(script_dir, "frozen_MIMIC",  "multiclass")
dir_mod_frozen_bin <- file.path(script_dir, "frozen_MIMIC",  "binary_OvR")
dir_mod_icht_mc    <- file.path(script_dir, "recal_ICHT",    "multiclass")
dir_mod_icht_bin   <- file.path(script_dir, "recal_ICHT",    "binary_OvR")
dir_mod_aumc_mc    <- file.path(script_dir, "recal_AUMC",    "multiclass")
dir_mod_aumc_bin   <- file.path(script_dir, "recal_AUMC",    "binary_OvR")
for (p in c(dir_mod_frozen_mc, dir_mod_frozen_bin,
            dir_mod_icht_mc,   dir_mod_icht_bin,
            dir_mod_aumc_mc,   dir_mod_aumc_bin)) dir.create(p, recursive = TRUE, showWarnings = FALSE)

class_levels <- paste0("C", 1:4)

# ---------------- 0.  Calibration helpers ----------------------------------
ece_brier <- function(prob, truth, k = 10) {
  cuts   <- seq(0, 1, length.out = k + 1)
  binfac <- factor(cut(prob, cuts, include.lowest = TRUE, labels = FALSE), levels = 1:k)
  exp    <- tapply(prob , binfac, mean,   na.rm = TRUE)
  obs    <- tapply(truth, binfac, mean,   na.rm = TRUE)
  counts <- tapply(truth, binfac, length)
  exp[is.na(exp)]       <- 0
  obs[is.na(obs)]       <- 0
  counts[is.na(counts)] <- 0
  ece_val   <- if (sum(counts) > 0) sum(abs(obs - exp) * counts) / sum(counts) else NA_real_
  brier_val <- mean((prob - truth)^2)
  df <- tibble(bin = seq_len(k), exp = exp, obs = obs, bin_count = counts)
  list(ece = ece_val, brier = brier_val, df = df)
}

# global theme: axis text/title size 18 everywhere
theme_font18 <- theme_bw(base_size = 18) +
  theme(axis.title = element_text(size = 18),
        axis.text  = element_text(size = 18),
        legend.title = element_text(size = 16),
        legend.text  = element_text(size = 14))

save_reliability <- function(cal_df, title, file_path) {
  p <- cal_df %>%
    dplyr::filter(bin_count > 0) %>%
    ggplot(aes(exp, obs, size = bin_count)) +
    geom_point(alpha = .7) +
    geom_abline(linetype = "dashed") +
    scale_size_continuous(range = c(1, 6)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = title, x = "Predicted", y = "Observed", size = "n") +
    theme_font18
  ggsave(filename = file_path, plot = p, width = 6, height = 6, dpi = 300)
}

# Other helper
# Always build a complete one-hot with the specified levels (even if some are absent)
onehot_levels <- function(y_factor, levels_out) {
  f <- factor(y_factor, levels = levels_out)
  M <- model.matrix(~ f - 1)
  colnames(M) <- paste0("class", levels_out)  # "classC1", "classC2", ...
  M
}

# Helper to fail fast if any stay lacks a class label
check_no_missing_class <- function(df, site, day, prob_tbl) {
  n_na <- sum(is.na(df$class))
  if (n_na == 0L) {
    message(sprintf("[Day %d %s] OK: no missing class labels (n=%d rows)", day, site, nrow(df)))
    return(invisible(TRUE))
  }
  # Diagnose: which stay_ids are missing, and whether they exist in the prob table
  miss_ids <- df %>% dplyr::filter(is.na(class)) %>% dplyr::pull(stay_id)
  in_prob  <- miss_ids %in% prob_tbl$stay_id
  
  diag_tbl <- tibble::tibble(
    stay_id = miss_ids,
    present_in_prob = in_prob
  )
  
  # Show a compact preview in console
  message(sprintf(
    "[Day %d %s] ERROR: %d rows have NA class. First few:\n%s",
    day, site, n_na, capture.output(print(head(diag_tbl, 20)))
  ))
  
  # Optional: write a CSV per day/site so you can inspect later
  if (exists("root_dir")) {
    out_csv <- file.path(root_dir, sprintf("missing_class_D%02d_%s.csv", day, site))
    readr::write_csv(diag_tbl, out_csv)
    message(sprintf("[Day %d %s] Missing-class list written to: %s", day, site, out_csv))
  }
  
  # Hard stop because coverage should be 100%
  stop(sprintf("[Day %d %s] Found %d stays without class labels. Fix the joins/prob table and rerun.",
               day, site, n_na))
}

# ---------------- 1.  Common objects ---------------------------------------
measure <- c("norad_vasorate","avg_lactate","locf_bicarbonate",
             "avg_peak_insp_pressure","avg_peep","avg_pao2fio2ratio",
             "locf_creatinine","avg_pco2","avg_minute_volume","avg_resp_rate")

# ---- Tuned hyperparameters from Day 3 tuning (2000 trials each) ----
# Multiclass: best mean accuracy = 0.265
param_mc  <- list(booster="gbtree", objective="multi:softprob",
                  eval_metric="mlogloss", 
                  eta=0.2, max_depth=8, min_child_weight=2,
                  subsample=0.6, colsample_bytree=1.0,
                  gamma=0.5, lambda=5, alpha=0)

# Binary classifiers: class-specific tuned parameters
# Default param_bin (used as base, will be overridden per class)
param_bin <- list(booster="gbtree", objective="binary:logistic",
                  eval_metric="auc", eta=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=1.0)

# Class-specific tuned binary parameters (from Day 3 tuning)
param_bin_tuned <- list(
  C1 = list(booster="gbtree", objective="binary:logistic", eval_metric="auc",
            eta=0.01, max_depth=8, min_child_weight=1,
            subsample=0.6, colsample_bytree=1.0,
            gamma=2, lambda=5, alpha=0),
  C2 = list(booster="gbtree", objective="binary:logistic", eval_metric="auc",
            eta=0.1, max_depth=3, min_child_weight=1,
            subsample=0.6, colsample_bytree=0.6,
            gamma=0, lambda=5, alpha=0),
  C3 = list(booster="gbtree", objective="binary:logistic", eval_metric="auc",
            eta=0.05, max_depth=3, min_child_weight=2,
            subsample=0.85, colsample_bytree=0.8,
            gamma=0.5, lambda=10, alpha=0),
  C4 = list(booster="gbtree", objective="binary:logistic", eval_metric="auc",
            eta=0.01, max_depth=5, min_child_weight=10,
            subsample=1.0, colsample_bytree=0.6,
            gamma=0, lambda=1, alpha=0.5)
)

# ---------------- 2.  MIMIC long → day-wide builder ------------------------
prob_mim  <- read_csv("Data/pprob_MIMIC.csv", show_col_types = FALSE) %>% select(stay_id, class)
long_mim <- read_csv("Data/mimic_dynamic_var.csv", show_col_types = FALSE) %>%
  mutate(
    age        = as.numeric(admission_age),
    gender_bin = if_else(gender == "M", 1, 0)
  )
static_mim <- long_mim %>% select(stay_id, age, gender_bin) %>% distinct()

wide_mim <- function(d) {
  long_mim %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = days_from_start,
      values_from= all_of(measure),
      names_glue = "day{days_from_start}_{.value}"
    ) %>%
    left_join(static_mim,  by = "stay_id") %>%
    left_join(prob_mim,   by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}

# ---------------- 3.  ICHT harmonisation builder ---------------------------
prob_icht <- read_csv("Data/pprob_ICHT.csv", show_col_types = FALSE) %>% select(stay_id, class)
long_icht <- read_csv("Data/icht_dynamic_var.csv", show_col_types = FALSE) %>%
  select(-c(avg_pco2, avg_pf_ratio)) %>%
  mutate(
    age        = as.numeric(age),
    gender_bin = if_else(gender %in% c("M", "Male", 1), 1, 0)
  )

# map already consistent column names to the common 'measure' set
icht_name_map <- c(
  days_from_start        = "day_from_start",
  norad_vasorate         = "avg_norad_equiv",
  avg_lactate            = "avg_lactate",
  locf_bicarbonate       = "avg_bicarb",
  avg_peak_insp_pressure = "avg_pip2",
  avg_peep               = "avg_peep",
  avg_pao2fio2ratio      = "avg_pf_ratio_mmHg",
  locf_creatinine        = "avg_creatinine",
  avg_pco2               = "avg_pco2_mmHg",
  avg_minute_volume      = "avg_minute_vent",
  avg_resp_rate          = "rr"
)

long_icht <- long_icht %>%
  rename(!!!icht_name_map) %>%
  mutate(avg_pao2fio2ratio = ifelse(avg_pao2fio2ratio > 470, NA, avg_pao2fio2ratio))

static_icht <- long_icht %>% select(stay_id, age, gender_bin) %>% distinct()

wide_icht <- function(d) {
  long_icht %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = days_from_start,
      values_from= all_of(measure),
      names_glue = "day{days_from_start}_{.value}"
    ) %>%
    left_join(static_icht, by = "stay_id") %>%
    left_join(prob_icht,  by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}

# ---------------- 3b.  AUMC harmonisation builder --------------------------
# Expected inputs:
#   - Data/pprob_AUMC.csv         with columns: stay_id (or admissionid), class
#   - Data/aumc_daily.csv         with daily variables (admissionid, day, ...)
#   - Data/umc_pf_daily.csv       with PF ratio by admission/day (stay_id, day_period, pf_ratio_avg)
#
# If your actual column names differ, tweak aumc_name_map and id merges below.
# ─────────────────────────────────────────────────────────────────────────────
# AUMC builder (uses age_imputed + weight_impute; norad_vasorate scaled)
# ─────────────────────────────────────────────────────────────────────────────

library(readr)
library(dplyr)
library(tidyr)
library(purrr)

# 1) Read inputs
prob_aumc_raw <- read_csv("Data/pprob_AUMC.csv", show_col_types = FALSE)
prob_aumc <- prob_aumc_raw %>%
  # standardise id column name to stay_id
  mutate(stay_id = dplyr::coalesce(.[["stay_id"]], .[["admissionid"]])) %>%
  select(stay_id, class)

long_aumc  <- read_csv("Data/aumc_daily.csv",   show_col_types = FALSE)
pf_aumc    <- read_csv("Data/umc_pf_daily.csv", show_col_types = FALSE)
demo_aumc  <- read_csv("Data/aumc_demo.csv",    show_col_types = FALSE)

# --- Standardise IDs in prob table to stay_id
if ("stay_id" %in% names(prob_aumc_raw)) {
  prob_aumc <- prob_aumc_raw %>% select(stay_id, class)
} else if ("admissionid" %in% names(prob_aumc_raw)) {
  prob_aumc <- prob_aumc_raw %>% rename(stay_id = admissionid) %>% select(stay_id, class)
} else {
  stop("pprob_AUMC.csv must contain either 'stay_id' or 'admissionid'.")
}

# --- Standardise IDs/time in aumc_daily
if (!"admissionid" %in% names(long_aumc) && "stay_id" %in% names(long_aumc)) {
  long_aumc <- long_aumc %>% rename(admissionid = stay_id)
}
if (!"day" %in% names(long_aumc) && "day_period" %in% names(long_aumc)) {
  long_aumc <- long_aumc %>% rename(day = day_period)
}
if (!all(c("admissionid","day") %in% names(long_aumc))) {
  stop("aumc_daily.csv must have 'admissionid' and 'day' (or mappable columns).")
}

# --- Standardise IDs/time in PF table
if (!"stay_id" %in% names(pf_aumc) && "admissionid" %in% names(pf_aumc)) {
  pf_aumc <- pf_aumc %>% rename(stay_id = admissionid)
}
if (!"day_period" %in% names(pf_aumc) && "day" %in% names(pf_aumc)) {
  pf_aumc <- pf_aumc %>% rename(day_period = day)
}
if (!all(c("stay_id","day_period") %in% names(pf_aumc))) {
  stop("umc_pf_daily.csv must have 'stay_id' and 'day_period' (or mappable columns).")
}

# --- Merge PF into daily long
long_aumc <- long_aumc %>%
  left_join(pf_aumc, by = c("admissionid" = "stay_id", "day" = "day_period"))

# --- Demo: ensure weight_impute exists; keep age_imputed
if (!"weight_impute" %in% names(demo_aumc)) {
  demo_aumc <- demo_aumc %>%
    mutate(weight_impute = if_else(!is.na(.data[["weight"]]), .data[["weight"]], 82.4))
}

# 4) Join demo to long and compute scaled norepinephrine
long_aumc <- long_aumc %>%
  left_join(demo_aumc %>%
              transmute(
                stay_id,
                sex,
                age_imputed = age_imputed,
                weight_impute = weight_impute
              ),
            by = c("admissionid" = "stay_id")) %>%
  mutate(
    # Use imputed age/sex for static features
    age_imputed      = as.numeric(age_imputed),
    gender_bin       = if_else(sex %in% c(1, "M", "Male"), 1, 0, missing = NA_real_),
    # Build vasorate in target units: mcg/kg/sec (norepinephrine typically mcg/min)
    norad_vasorate   = ifelse(is.na(weight_impute) | weight_impute <= 0,
                              NA_real_,
                              .data[["norepinephrine"]] / weight_impute)
  )

# 5) Map AUMC columns to the shared 'measure' names
#    (we already created norad_vasorate above, so map it to itself)
aumc_name_map <- c(
  # IDs / time
  stay_id                = "admissionid",
  days_from_start        = "day",
  # measures
  norad_vasorate         = "norad_vasorate",
  avg_lactate            = "lactate",
  locf_bicarbonate       = "bicarb_mmol_l",
  avg_peak_insp_pressure = "ppeak_cmh2o",
  avg_peep               = "peep_cmh2o",
  avg_pao2fio2ratio      = "pf_ratio_avg",
  locf_creatinine        = "creatinine_mg_dl",
  avg_pco2               = "pco2_mmhg",
  avg_minute_volume      = "minute_vent",
  avg_resp_rate          = "resp_rate_bpm"
)

long_aumc <- long_aumc %>%
  rename(!!!aumc_name_map) %>%
  mutate(
    age        = age_imputed,                               # <- use imputed age
    avg_pao2fio2ratio = ifelse(avg_pao2fio2ratio > 470, NA, avg_pao2fio2ratio)
  )

# 6) Static table for AUMC
static_aumc <- long_aumc %>%
  select(stay_id, age, gender_bin) %>%
  distinct()

# 7) Wide builder
wide_aumc <- function(d) {
  long_aumc %>%
    filter(days_from_start %in% 0:d) %>%
    tidyr::pivot_wider(
      id_cols    = stay_id,
      names_from = days_from_start,
      values_from= all_of(measure),
      names_glue = "day{days_from_start}_{.value}"
    ) %>%
    left_join(static_aumc, by = "stay_id") %>%
    left_join(prob_aumc,   by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}


# ── Remove AUMC stays that lack a class label (hard delete for this run) ──
aumc_missing_ids <- long_aumc %>%
  dplyr::distinct(stay_id) %>%
  dplyr::anti_join(prob_aumc, by = "stay_id") %>%
  dplyr::pull(stay_id)

if (length(aumc_missing_ids) > 0L) {
  message(sprintf("[AUMC] Removing %d stays without class labels. First few: %s",
                  length(aumc_missing_ids),
                  paste(head(aumc_missing_ids, 10), collapse = ", ")))
  # Drop from long/static to prevent leakage into any day window
  long_aumc   <- long_aumc   %>% dplyr::filter(!stay_id %in% aumc_missing_ids)
  static_aumc <- static_aumc %>% dplyr::filter(!stay_id %in% aumc_missing_ids)
  
  # Optional: save the list for audit
  if (exists("root_dir")) {
    readr::write_csv(
      tibble::tibble(stay_id = aumc_missing_ids),
      file.path(root_dir, "AUMC_removed_missing_class.csv")
    )
  }
} else {
  message("[AUMC] OK: all stays have class labels.")
}

# ─────────────────────────────────────────────────────────────────────────────
# QUICK INSPECTIONS: Compare distributions & units across cohorts
# (run after long_mim, long_icht already exist and after the renaming above)
# ─────────────────────────────────────────────────────────────────────────────

inspect_vars <- c("norad_vasorate","avg_lactate","locf_bicarbonate",
                  "avg_peak_insp_pressure","avg_peep","avg_pao2fio2ratio",
                  "locf_creatinine","avg_pco2","avg_minute_volume","avg_resp_rate")

quick_describe <- function(df, site){
  df %>%
    summarise(
      across(all_of(inspect_vars),
             list(n = ~sum(!is.na(.)),
                  miss_pct = ~mean(is.na(.))*100,
                  mean = ~mean(., na.rm = TRUE),
                  sd = ~sd(., na.rm = TRUE),
                  min = ~suppressWarnings(min(., na.rm = TRUE)),
                  max = ~suppressWarnings(max(., na.rm = TRUE))),
             .names = "{.col}__{.fn}")
    ) %>%
    pivot_longer(everything()) %>%
    separate(name, into = c("variable","stat"), sep = "__") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(site = site, .before = 1)
}

cat("\n\n==== QUICK DISTRIBUTION CHECKS ====\n")

desc_mim  <- quick_describe(long_mim  %>% rename(days_from_start = days_from_start), "MIMIC")
desc_icht <- quick_describe(long_icht %>% rename(days_from_start = days_from_start), "ICHT")
desc_aumc <- quick_describe(long_aumc %>% rename(days_from_start = days_from_start), "AUMC")

dist_all <- bind_rows(desc_mim, desc_icht, desc_aumc) %>%
  arrange(variable, site)

print(dist_all, n = 200)

cat("\n\n---- Norepinephrine scaling spot-check (AUMC) ----\n")
ne_check <- long_aumc %>%
  transmute(stay_id, day = days_from_start,
            norepinephrine_raw = !!rlang::sym("norepinephrine"),  # raw column still in long_aumc?
            weight_impute,
            norad_vasorate) %>%
  filter(!is.na(norepinephrine_raw) & !is.na(weight_impute)) %>%
  slice_head(n = 12)
print(ne_check)

cat("\n\n==== DONE: AUMC builder updated; distributions printed ====\n")



# ─────────────────────────────────────────────────────────────────────────────
# PATCH: unit conversions + clipping (run after long_mim/long_icht/long_aumc exist)
# ─────────────────────────────────────────────────────────────────────────────
# 1) Non-negative-only vars: set negatives to NA
fix_nonneg <- function(df, vars){
  for (v in intersect(vars, names(df))) {
    df[[v]] <- ifelse(df[[v]] < 0, NA_real_, df[[v]])
  }
  df
}

nn_vars <- c("avg_lactate","avg_minute_volume","avg_peep","avg_peak_insp_pressure",
             "avg_resp_rate","norad_vasorate","locf_bicarbonate","locf_creatinine")
long_mim  <- fix_nonneg(long_mim,  nn_vars)
long_icht <- fix_nonneg(long_icht, nn_vars)
long_aumc <- fix_nonneg(long_aumc, nn_vars)

# 2) Minute ventilation: convert obvious mL/min to L/min
convert_minute_vent <- function(df) {
  if (!"avg_minute_volume" %in% names(df)) return(df)
  df$avg_minute_volume <- ifelse(df$avg_minute_volume > 200,
                                 df$avg_minute_volume/1000, df$avg_minute_volume)
  df
}
long_mim  <- convert_minute_vent(long_mim)
long_icht <- convert_minute_vent(long_icht)
long_aumc <- convert_minute_vent(long_aumc)

if ("locf_creatinine" %in% names(long_icht)) {
  long_icht <- long_icht %>%
    mutate(locf_creatinine = locf_creatinine / 88.4)   # <-- this is the change
}

# 3) Physiology-based clipping (keeps tails but squashes implausible)
clip_between <- function(x, lo, hi) ifelse(is.na(x), NA_real_, pmax(pmin(x, hi), lo))

for (nm in c("avg_lactate","avg_pco2","avg_peak_insp_pressure","avg_peep",
             "avg_resp_rate","locf_bicarbonate","avg_pao2fio2ratio","norad_vasorate")) {
  for (DF in c("long_mim","long_icht","long_aumc")) {
    df <- get(DF)
    if (!nm %in% names(df)) next
    df[[nm]] <- switch(nm,
                       avg_lactate            = clip_between(df[[nm]], 0,   20),
                       avg_pco2               = clip_between(df[[nm]], 10, 120),
                       avg_peak_insp_pressure = clip_between(df[[nm]], 0,   80),
                       avg_peep               = clip_between(df[[nm]], 0,   30),
                       avg_resp_rate          = clip_between(df[[nm]], 0,   80),
                       locf_bicarbonate       = clip_between(df[[nm]], 5,   50),
                       avg_pao2fio2ratio      = clip_between(df[[nm]], 30, 470),
                       norad_vasorate         = pmax(df[[nm]], 0),
                       df[[nm]]
    )
    assign(DF, df)
  }
}

# 4) Creatinine spot-check (should be ~0.5–3 mg/dL typically; tails ok)
creat_check <- function(df, site) {
  x <- df$locf_creatinine
  data.frame(
    site = site, n = sum(!is.na(x)),
    med = median(x, na.rm=TRUE),
    p05 = quantile(x, .05, na.rm=TRUE),
    p95 = quantile(x, .95, na.rm=TRUE),
    lt_0p3_pct = mean(x < 0.3, na.rm=TRUE)*100,
    gt_5_pct   = mean(x > 5,   na.rm=TRUE)*100
  )
}
print(dplyr::bind_rows(
  creat_check(long_mim,  "MIMIC"),
  creat_check(long_icht, "ICHT"),
  creat_check(long_aumc, "AUMC")
))
# ─────────────────────────────────────────────────────────────────────────────
# Re-run the quick distribution comparison to verify the cleanup
# ─────────────────────────────────────────────────────────────────────────────
inspect_vars <- c("norad_vasorate","avg_lactate","locf_bicarbonate",
                  "avg_peak_insp_pressure","avg_peep","avg_pao2fio2ratio",
                  "locf_creatinine","avg_pco2","avg_minute_volume","avg_resp_rate")

quick_describe <- function(df, site){
  df %>%
    summarise(
      across(all_of(inspect_vars),
             list(n = ~sum(!is.na(.)),
                  miss_pct = ~mean(is.na(.))*100,
                  mean = ~mean(., na.rm = TRUE),
                  sd = ~sd(., na.rm = TRUE),
                  min = ~suppressWarnings(min(., na.rm = TRUE)),
                  max = ~suppressWarnings(max(., na.rm = TRUE))),
             .names = "{.col}__{.fn}")
    ) %>%
    tidyr::pivot_longer(everything()) %>%
    tidyr::separate(name, into = c("variable","stat"), sep = "__") %>%
    tidyr::pivot_wider(names_from = stat, values_from = value) %>%
    mutate(site = site, .before = 1)
}

cat("\n\n==== DISTRIBUTIONS AFTER CONVERSION + CLIPPING ====\n")
desc_mim2  <- quick_describe(long_mim,  "MIMIC")
desc_icht2 <- quick_describe(long_icht, "ICHT")
desc_aumc2 <- quick_describe(long_aumc, "AUMC")

dist_all2 <- dplyr::bind_rows(desc_mim2, desc_icht2, desc_aumc2) %>%
  arrange(variable, site)
print(dist_all2, n = 200)

cat("\n\n---- Spot-checks ----\n")
cat("* ICHT creatinine should now be mg/dL (means ~1–2):\n")
print(dist_all2 %>% dplyr::filter(variable == "locf_creatinine"))

cat("\n* Minute ventilation should now be ~5–20 L/min range (no ultra-high maxima):\n")
print(dist_all2 %>% dplyr::filter(variable == "avg_minute_volume"))




suppressPackageStartupMessages({ library(dplyr); library(purrr); library(tidyr); library(ggplot2); library(caret); library(pROC); library(readr); library(stringr); })

# ---- Threshold helpers ----
threshold_for_specificity <- function(y01, p, spec_target = 0.90){
  ord <- order(p); thr <- c(-Inf, unique(p[ord]), Inf)
  spec <- vapply(thr, function(t){
    pred <- as.integer(p > t)
    tn <- sum(pred==0 & y01==0); fp <- sum(pred==1 & y01==0)
    if ((tn+fp)==0) return(NA_real_); tn/(tn+fp)
  }, numeric(1))
  i <- which(spec >= spec_target)[1]
  if (is.na(i)) return(0.5) else thr[i]
}

threshold_youden <- function(y01, p){
  ord <- order(p); thr <- c(-Inf, unique(p[ord]), Inf)
  yj <- vapply(thr, function(t){
    pred <- as.integer(p > t)
    tp <- sum(pred==1 & y01==1); fn <- sum(pred==0 & y01==1)
    tn <- sum(pred==0 & y01==0); fp <- sum(pred==1 & y01==0)
    sens <- ifelse((tp+fn)>0, tp/(tp+fn), NA_real_)
    spec <- ifelse((tn+fp)>0, tn/(tn+fp), NA_real_)
    if (is.na(sens) || is.na(spec)) return(-Inf)
    sens + spec - 1
  }, numeric(1))
  thr[ which.max(yj) ][1] %||% 0.5
}

`%||%` <- function(x, y) if (is.null(x) || length(x)==0 || is.na(x)) y else x

# ---- Run binary metrics for one class & probabilities at a given threshold ----
bin_metrics_at <- function(prob, y01, thr, cls, day){
  pred01 <- as.integer(prob > thr)
  truth  <- factor(y01, levels = c(0,1))
  pred   <- factor(pred01, levels = c(0,1))
  roc_o  <- pROC::roc(y01, prob, quiet = TRUE)
  cm_b   <- caret::confusionMatrix(pred, truth, positive = "1")
  cal_b  <- ece_brier(prob, y01)
  tibble(day = day, class = cls,
         AUC = as.numeric(pROC::auc(roc_o)),
         Accuracy = cm_b$overall["Accuracy"],
         Sensitivity = cm_b$byClass["Sensitivity"],
         Specificity = cm_b$byClass["Specificity"],
         Brier = cal_b$brier, ECE = cal_b$ece)
}

# ---- Store calibration bins for panel plots (independent of threshold) ----
collect_cal_bins <- function(prob, y01, cls, day, condition){
  cal <- ece_brier(prob, y01)
  cal$df %>%
    mutate(class = cls, day = day, condition = condition) %>%
    select(condition, day, class, bin, exp, obs, bin_count)
}

# ---- Helper: safely extract best_iteration from xgb.cv (handles xgboost 3.x) ----
# xgboost 3.x: best_iteration moved to cv$early_stop$best_iteration
safe_best_iter <- function(cv, default = 50) {
  iter <- cv$early_stop$best_iteration           # xgboost 3.x
  if (is.null(iter) || length(iter) == 0) iter <- cv$best_iteration    # older versions
  if (is.null(iter) || length(iter) == 0) iter <- cv$best_ntreelimit   # even older
  if (is.null(iter) || length(iter) == 0) iter <- nrow(cv$evaluation_log)
  if (is.null(iter) || length(iter) == 0 || iter == 0) iter <- default
  return(as.integer(iter))
}

# ---- Helper: safely run xgb.cv with fallback to default nrounds ----
safe_xgb_cv <- function(params, data, nrounds = 300, nfold = 5, 
                        early_stopping_rounds = 20, verbose = 0, 
                        default_rounds = 50, label = "") {
  result <- tryCatch({
    xgb.cv(params, data, nrounds = nrounds, nfold = nfold,
           early_stopping_rounds = early_stopping_rounds, verbose = verbose)
  }, error = function(e) {
    message(sprintf("[%s] xgb.cv failed (using %d rounds): %s", label, default_rounds, e$message))
    NULL
  })
  return(result)
}


# Multiclass
frozen_mc_mimic <- tibble()
primary_mc_icht <- tibble(); primary_mc_aumc <- tibble()
recal_mc_icht   <- tibble(); recal_mc_aumc  <- tibble()

# Binary – MIMIC holdout (three operating points)
primary_bin_mimic_t05  <- tibble()
primary_bin_mimic_spec <- tibble()
primary_bin_mimic_yj   <- tibble()

# Binary – ICHT/AUMC frozen (MIMIC model) at T05/SPEC/YJ
primary_bin_icht_t05 <- tibble(); primary_bin_icht_spec <- tibble(); primary_bin_icht_yj <- tibble()
primary_bin_aumc_t05 <- tibble(); primary_bin_aumc_spec <- tibble(); primary_bin_aumc_yj <- tibble()

# Binary – recal (site-trained) with fixed 0.5 and SPEC
recal_bin_icht <- tibble(); recal_bin_aumc <- tibble()
recal_bin_icht_spec <- tibble(); recal_bin_aumc_spec <- tibble()

# Calibration stores (facetable)
cal_mimic_bin       <- tibble()
cal_icht_frozen_bin <- tibble()
cal_icht_recal_bin  <- tibble()
cal_aumc_frozen_bin <- tibble()
cal_aumc_recal_bin  <- tibble()

# Feature importance stores (for multiclass and binary models)
fi_multiclass <- tibble()
fi_binary     <- tibble()

class_levels <- paste0("C", 1:4)  # if not already set


# ---------------- 4b. Day-3 hyperparameter tuning --------------------------
# Goal: tune multiclass + OvR binary models for day 3,
#       selecting params that generalize to MIMIC holdout + ICHT + AUMC.
# Set to FALSE to skip the tuning step.
# NOTE: Tuning already completed - results saved in Hyperparameter_tuning_day3/
#       Best parameters are now hardcoded in param_mc and param_bin_tuned above.
run_day3_tuning <- FALSE

if (run_day3_tuning) {
  suppressPackageStartupMessages({
    library(future); library(furrr)
  })
  # Parallel setup (Windows-safe): use multisession
  max_workers <- 10
  plan(multisession, workers = min(availableCores(), max_workers))
  on.exit(plan(sequential), add = TRUE)

  tune_dir <- file.path(root_dir, "Hyperparameter_tuning_day3")
  dir.create(tune_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(101)
  d_tune <- 3
  
  # Build day-3 datasets
  mim_t  <- wide_mim(d_tune)
  icht_t <- wide_icht(d_tune)
  aumc_t <- wide_aumc(d_tune)
  preds_t <- setdiff(names(mim_t), c("stay_id","class"))
  
  # Guards
  check_no_missing_class(mim_t,  "MIMIC", d_tune, prob_mim)
  check_no_missing_class(icht_t, "ICHT",  d_tune, prob_icht)
  check_no_missing_class(aumc_t, "AUMC",  d_tune, prob_aumc)
  
  message("[Tuning] Day 3 datasets ready: MIMIC=", nrow(mim_t), ", ICHT=", nrow(icht_t), ", AUMC=", nrow(aumc_t))
  message("[Tuning] Features: ", length(preds_t))
  
  # MIMIC split (holdout); external uses all labeled rows
  idx_m <- createDataPartition(mim_t$class, p = .8, list = FALSE)
  tr_m_t <- mim_t[idx_m, , drop = FALSE]
  te_m_t <- mim_t[-idx_m, , drop = FALSE]
  ext_i  <- icht_t %>% dplyr::filter(!is.na(class))
  ext_a  <- aumc_t %>% dplyr::filter(!is.na(class))
  
  safe_auc <- function(y01, p){
    if (length(unique(y01[!is.na(y01)])) < 2) return(NA_real_)
    as.numeric(pROC::auc(pROC::roc(y01, p, quiet = TRUE)))
  }
  
  # Random parameter sampler
  sample_param_grid <- function(n) {
    tibble(
      eta = sample(c(0.01, 0.03, 0.05, 0.1, 0.2), n, replace = TRUE),
      max_depth = sample(c(3, 4, 5, 6, 8), n, replace = TRUE),
      min_child_weight = sample(c(1, 2, 5, 10), n, replace = TRUE),
      subsample = sample(c(0.6, 0.7, 0.85, 1.0), n, replace = TRUE),
      colsample_bytree = sample(c(0.6, 0.8, 1.0), n, replace = TRUE),
      gamma = sample(c(0, 0.5, 1, 2), n, replace = TRUE),
      lambda = sample(c(0, 1, 5, 10), n, replace = TRUE),
      alpha = sample(c(0, 0.5, 1), n, replace = TRUE)
    )
  }
  
  # -------- Multiclass tuning --------
  eval_mc_params <- function(eta, max_depth, min_child_weight, subsample,
                             colsample_bytree, gamma, lambda, alpha) {
    param_mc_t <- list(
      booster = "gbtree",
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      nthread = 1,
      num_class = length(class_levels),
      eta = eta,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      lambda = lambda,
      alpha = alpha
    )
    dmat_tr <- xgb.DMatrix(as.matrix(tr_m_t[, preds_t]), label = as.numeric(tr_m_t$class) - 1)
    cv <- xgb.cv(param_mc_t, dmat_tr, nrounds = 400, nfold = 5,
                 early_stopping_rounds = 20, verbose = 0)
    # Handle best_iteration - new xgboost versions may use different accessor
    best_iter <- safe_best_iter(cv)
    if (is.null(best_iter) || length(best_iter) == 0 || best_iter == 0) best_iter <- 50  # fallback
    bst <- xgb.train(param_mc_t, dmat_tr, nrounds = best_iter, verbose = 0)
    
    pred_m <- predict(bst, xgb.DMatrix(as.matrix(te_m_t[, preds_t]))) %>%
      matrix(ncol = length(class_levels), byrow = TRUE)
    pred_i <- predict(bst, xgb.DMatrix(as.matrix(ext_i[, preds_t]))) %>%
      matrix(ncol = length(class_levels), byrow = TRUE)
    pred_a <- predict(bst, xgb.DMatrix(as.matrix(ext_a[, preds_t]))) %>%
      matrix(ncol = length(class_levels), byrow = TRUE)
    
    acc_m <- confusionMatrix(factor(class_levels[max.col(pred_m)], levels = class_levels), te_m_t$class)$overall["Accuracy"]
    acc_i <- confusionMatrix(factor(class_levels[max.col(pred_i)], levels = class_levels), ext_i$class)$overall["Accuracy"]
    acc_a <- confusionMatrix(factor(class_levels[max.col(pred_a)], levels = class_levels), ext_a$class)$overall["Accuracy"]
    
    y_m <- onehot_levels(te_m_t$class, class_levels)
    y_i <- onehot_levels(ext_i$class, class_levels)
    y_a <- onehot_levels(ext_a$class, class_levels)
    
    brier_m <- mean((pred_m - y_m)^2)
    brier_i <- mean((pred_i - y_i)^2)
    brier_a <- mean((pred_a - y_a)^2)
    
    tibble(
      eta = eta, max_depth = max_depth, min_child_weight = min_child_weight,
      subsample = subsample, colsample_bytree = colsample_bytree,
      gamma = gamma, lambda = lambda, alpha = alpha,
      best_iteration = best_iter,
      acc_mimic = as.numeric(acc_m), acc_icht = as.numeric(acc_i), acc_aumc = as.numeric(acc_a),
      brier_mimic = brier_m, brier_icht = brier_i, brier_aumc = brier_a,
      score_mean_acc = mean(c(acc_m, acc_i, acc_a), na.rm = TRUE),
      score_mean_brier = mean(c(brier_m, brier_i, brier_a), na.rm = TRUE)
    )
  }
  
  # Adjust these to test ~10k–20k models (across MC + per-class binary)
  # NOTE: Set mc_trials=2000 and bin_trials=2000 for full tuning; 
  #       using smaller values for testing
  mc_trials <- 2000  # Full run
  mc_grid <- sample_param_grid(mc_trials)
  message("[Tuning] Starting multiclass hyperparameter tuning with ", mc_trials, " trials...")
  message("[Tuning] Using ", nbrOfWorkers(), " parallel workers")
  flush.console()
  
  mc_results <- tryCatch({
    res <- furrr::future_pmap_dfr(mc_grid, eval_mc_params,
                           .options = furrr::furrr_options(seed = TRUE),
                           .progress = TRUE)
    message("[Tuning] Multiclass tuning completed, got ", nrow(res), " rows")
    res
  }, error = function(e) {
    message("[Tuning ERROR] Multiclass tuning failed: ", conditionMessage(e))
    NULL
  })
  
  if (!is.null(mc_results) && nrow(mc_results) > 0) {
    mc_results <- mc_results %>% arrange(desc(score_mean_acc), score_mean_brier)
    outfile_mc <- file.path(tune_dir, "day3_multiclass_tuning_results.csv")
    write_csv(mc_results, outfile_mc)
    message("[Tuning] Saved multiclass results: ", nrow(mc_results), " rows to ", outfile_mc)
    message("[Tuning] Best multiclass mean accuracy: ", round(mc_results$score_mean_acc[1], 4))
  } else {
    message("[Tuning] No multiclass results to save")
  }
  
  # -------- Binary OvR tuning (per class) --------
  eval_bin_params <- function(eta, max_depth, min_child_weight, subsample,
                              colsample_bytree, gamma, lambda, alpha, cls) {
    ytr <- as.numeric(tr_m_t$class == cls)
    if (length(unique(ytr)) < 2) return(NULL)
    spw <- sum(ytr == 0) / sum(ytr == 1)
    param_bin_t <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      nthread = 1,
      scale_pos_weight = spw,
      eta = eta,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      gamma = gamma,
      lambda = lambda,
      alpha = alpha
    )
    dmat_tr <- xgb.DMatrix(as.matrix(tr_m_t[, preds_t]), label = ytr)
    cv <- xgb.cv(param_bin_t, dmat_tr, nrounds = 400, nfold = 5,
                 early_stopping_rounds = 20, verbose = 0)
    # Handle best_iteration - new xgboost versions may use different accessor
    best_iter <- safe_best_iter(cv)
    if (is.null(best_iter) || length(best_iter) == 0 || best_iter == 0) best_iter <- 50  # fallback
    bst <- xgb.train(param_bin_t, dmat_tr, nrounds = best_iter, verbose = 0)
    
    pm <- predict(bst, xgb.DMatrix(as.matrix(te_m_t[, preds_t])))
    pi <- predict(bst, xgb.DMatrix(as.matrix(ext_i[, preds_t])))
    pa <- predict(bst, xgb.DMatrix(as.matrix(ext_a[, preds_t])))
    
    auc_m <- safe_auc(as.numeric(te_m_t$class == cls), pm)
    auc_i <- safe_auc(as.numeric(ext_i$class == cls), pi)
    auc_a <- safe_auc(as.numeric(ext_a$class == cls), pa)
    
    tibble(
      class = cls,
      eta = eta, max_depth = max_depth, min_child_weight = min_child_weight,
      subsample = subsample, colsample_bytree = colsample_bytree,
      gamma = gamma, lambda = lambda, alpha = alpha,
      best_iteration = best_iter,
      auc_mimic = auc_m, auc_icht = auc_i, auc_aumc = auc_a,
      score_mean_auc = mean(c(auc_m, auc_i, auc_a), na.rm = TRUE)
    )
  }
  
  bin_trials <- 2000  # Full run
  bin_grid <- sample_param_grid(bin_trials)
  
  message("[Tuning] Starting binary (OvR) hyperparameter tuning with ", bin_trials, " trials per class...")
  flush.console()
  for (cls in class_levels) {
    message("[Tuning] Binary tuning for class ", cls, "...")
    bin_grid_cls <- bin_grid %>% mutate(cls = cls)
    bin_results <- tryCatch({
      furrr::future_pmap_dfr(bin_grid_cls, eval_bin_params,
                             .options = furrr::furrr_options(seed = TRUE),
                             .progress = TRUE)
    }, error = function(e) {
      message("[Tuning ERROR] Binary tuning for ", cls, " failed: ", conditionMessage(e))
      NULL
    })
    if (is.null(bin_results) || nrow(bin_results) == 0L) {
      message("[Tuning] No results for class ", cls)
      next
    }
    bin_results <- bin_results %>% arrange(desc(score_mean_auc))
    write_csv(bin_results, file.path(tune_dir, sprintf("day3_binary_tuning_results_%s.csv", cls)))
    message("[Tuning] Saved binary results for ", cls, ": ", nrow(bin_results), " rows, best AUC: ", round(bin_results$score_mean_auc[1], 4))
  }
  
  message("[Tuning] Day 3 hyperparameter tuning complete!")
}


# =============================== MAIN LOOP =================================
for (d in 0:7){
  message(">>>>  Day window 0-", d, "  <<<<")
  
  # Build matrices
  mim   <- wide_mim(d)
  icht  <- wide_icht(d)
  aumc  <- wide_aumc(d)
  preds <- setdiff(names(mim), c("stay_id","class"))
  
  # Guards
  check_no_missing_class(mim,  "MIMIC", d, prob_mim)
  check_no_missing_class(icht, "ICHT",  d, prob_icht)
  check_no_missing_class(aumc, "AUMC",  d, prob_aumc)
  
  # ===== MIMIC train (frozen) =====
  set.seed(1 + d)
  idx_m <- createDataPartition(mim$class, p = .8, list = FALSE)
  tr_m  <- mim[idx_m,];  te_m <- mim[-idx_m,]
  
  param_mc_m <- c(param_mc, list(num_class = length(levels(tr_m$class))))
  dmat_m_mc  <- xgb.DMatrix(as.matrix(tr_m[, preds]), label = as.numeric(tr_m$class) - 1)
  cv_m_mc    <- safe_xgb_cv(param_mc_m, dmat_m_mc, nrounds = 300, nfold = 5,
                            early_stopping_rounds = 20, verbose = 0, label = paste0("Day ", d, " MIMIC MC"))
  mc_m       <- xgb.train(param_mc_m, dmat_m_mc, nrounds = safe_best_iter(cv_m_mc, default = 50), verbose = 0)
  xgb.save(mc_m, file.path(dir_mod_frozen_mc, sprintf("day%d_multiclass.json", d)))

  # Extract multiclass feature importance
  imp_mc <- xgb.importance(feature_names = preds, model = mc_m)
  fi_multiclass <<- bind_rows(fi_multiclass,
                              imp_mc %>% mutate(day = d, model = "multiclass", site = "MIMIC"))

  # MIMIC multiclass holdout (ref)
  pmat_mimic <- predict(mc_m, xgb.DMatrix(as.matrix(te_m[, preds]))) %>% matrix(ncol = length(class_levels), byrow = TRUE)
  acc_mimic  <- confusionMatrix(factor(class_levels[max.col(pmat_mimic)], levels = class_levels), te_m$class)$overall["Accuracy"]
  frozen_mc_mimic <- bind_rows(frozen_mc_mimic, tibble(day = d, Accuracy = acc_mimic))
  
  # MIMIC binary OvR (class-weighted, using tuned hyperparameters)
  bin_m <- map(setNames(levels(tr_m$class), levels(tr_m$class)), \(cls){
    y <- as.numeric(tr_m$class == cls)
    if (length(unique(y)) < 2) { warning(sprintf("[Day %d MIMIC] skip OvR %s", d, cls)); return(NULL) }
    spw <- sum(y == 0) / sum(y == 1)
    # Use class-specific tuned parameters if available
    base_params <- if (!is.null(param_bin_tuned[[cls]])) param_bin_tuned[[cls]] else param_bin
    param_bin_w <- modifyList(base_params, list(scale_pos_weight = spw))
    dmat <- xgb.DMatrix(as.matrix(tr_m[, preds]), label = y)
    cvb <- tryCatch({
      xgb.cv(param_bin_w, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
    }, error = function(e) {
      message(sprintf("[Day %d %s] xgb.cv failed, using default 50 rounds: %s", d, cls, e$message))
      NULL
    })
    nrounds_use <- if (!is.null(cvb)) safe_best_iter(cvb) else 50
    xgb.train(param_bin_w, dmat, nrounds = nrounds_use, verbose = 0)
  })
  # Save frozen MIMIC binary OvR models
  for (cls in class_levels) {
    if (!is.null(bin_m[[cls]]))
      xgb.save(bin_m[[cls]], file.path(dir_mod_frozen_bin, sprintf("day%d_%s.json", d, cls)))
  }

  # Extract binary feature importance for each class
  for (cls in class_levels) {
    if (!is.null(bin_m[[cls]])) {
      imp_bin <- xgb.importance(feature_names = preds, model = bin_m[[cls]])
      fi_binary <<- bind_rows(fi_binary,
                              imp_bin %>% mutate(day = d, model = paste0("binary_", cls),
                                                 class = cls, site = "MIMIC"))
    }
  }

  # Learn SPEC & YJ thresholds on MIMIC holdout
  spec_targets <- c(C1=0.90, C2=0.95, C3=0.90, C4=0.90)
  tau_spec <- setNames(vector("list", length(class_levels)), class_levels)
  tau_yj   <- setNames(vector("list", length(class_levels)), class_levels)
  
  for (cls in class_levels) {
    if (is.null(bin_m[[cls]])) { tau_spec[[cls]] <- 0.5; tau_yj[[cls]] <- 0.5; next }
    prob_m <- predict(bin_m[[cls]], xgb.DMatrix(as.matrix(te_m[, preds])))
    y_m    <- as.numeric(te_m$class == cls)
    tau_spec[[cls]] <- threshold_for_specificity(y_m, prob_m, spec_target = spec_targets[[cls]])
    tau_yj[[cls]]   <- threshold_youden(y_m, prob_m)
    # collect calibration for MIMIC (doesn't depend on threshold)
    cal_mimic_bin <- bind_rows(cal_mimic_bin, collect_cal_bins(prob_m, y_m, cls, d, "MIMIC"))
  }
  
  # Save MIMIC holdout metrics at T05, SPEC, YJ
  for (cls in class_levels) {
    if (is.null(bin_m[[cls]])) {
      primary_bin_mimic_t05  <- bind_rows(primary_bin_mimic_t05,  tibble(day=d, class=cls, AUC=NA, Accuracy=NA, Sensitivity=NA, Specificity=NA, Brier=NA, ECE=NA))
      primary_bin_mimic_spec <- bind_rows(primary_bin_mimic_spec, tibble(day=d, class=cls, AUC=NA, Accuracy=NA, Sensitivity=NA, Specificity=NA, Brier=NA, ECE=NA))
      primary_bin_mimic_yj   <- bind_rows(primary_bin_mimic_yj,   tibble(day=d, class=cls, AUC=NA, Accuracy=NA, Sensitivity=NA, Specificity=NA, Brier=NA, ECE=NA))
      next
    }
    prob_m <- predict(bin_m[[cls]], xgb.DMatrix(as.matrix(te_m[, preds])))
    y_m    <- as.numeric(te_m$class == cls)
    
    primary_bin_mimic_t05  <- bind_rows(primary_bin_mimic_t05,  bin_metrics_at(prob_m, y_m, 0.5,             cls, d))
    primary_bin_mimic_spec <- bind_rows(primary_bin_mimic_spec, bin_metrics_at(prob_m, y_m, tau_spec[[cls]],  cls, d))
    primary_bin_mimic_yj   <- bind_rows(primary_bin_mimic_yj,   bin_metrics_at(prob_m, y_m, tau_yj[[cls]],    cls, d))
  }
  
  # ===== Split ICHT / AUMC (same split for frozen/recal) =====
  set.seed(2 + d); idx_i <- createDataPartition(icht$class, p = .8, list = FALSE); tr_i <- icht[idx_i,]; te_i <- icht[-idx_i,]
  set.seed(3 + d); idx_a <- createDataPartition(aumc$class, p = .8, list = FALSE); tr_a <- aumc[idx_a,]; te_a <- aumc[-idx_a,]
  
  # ===== PRIMARY (frozen MIMIC → ICHT/AUMC) =====
  for (site in c("ICHT","AUMC")) {
    te <- if (site=="ICHT") te_i else te_a
    keep <- !is.na(te$class); te <- te[keep, , drop = FALSE]; if (nrow(te)==0L) next
    
    # Multiclass (frozen)
    pmat_frozen <- predict(mc_m, xgb.DMatrix(as.matrix(te[, preds]))) %>% matrix(ncol = length(class_levels), byrow = TRUE)
    Y_onehot_f  <- onehot_levels(te$class, class_levels)
    cm_mc_f     <- confusionMatrix(factor(class_levels[max.col(pmat_frozen)], levels = class_levels), te$class)
    row <- tibble(day = d, Accuracy = cm_mc_f$overall["Accuracy"],
                  Brier = mean((pmat_frozen - Y_onehot_f)^2),
                  ECE   = ece_brier(as.numeric(pmat_frozen), as.numeric(Y_onehot_f))$ece)
    if (site=="ICHT") primary_mc_icht <- bind_rows(primary_mc_icht, row) else primary_mc_aumc <- bind_rows(primary_mc_aumc, row)
    
    # Binary (frozen) at T05/SPEC/YJ using MIMIC thresholds and probabilities
    for (cls in class_levels) {
      if (is.null(bin_m[[cls]])) next
      prob  <- predict(bin_m[[cls]], xgb.DMatrix(as.matrix(te[, preds])))
      yte   <- as.numeric(te$class == cls)
      # collect calibration for frozen external
      if (site=="ICHT") cal_icht_frozen_bin <- bind_rows(cal_icht_frozen_bin, collect_cal_bins(prob, yte, cls, d, "ICHT_frozen"))
      if (site=="AUMC") cal_aumc_frozen_bin <- bind_rows(cal_aumc_frozen_bin, collect_cal_bins(prob, yte, cls, d, "AUMC_frozen"))
      
      # T05
      m_t05 <- bin_metrics_at(prob, yte, 0.5, cls, d)
      # SPEC
      m_spc <- bin_metrics_at(prob, yte, tau_spec[[cls]], cls, d)
      # YJ (MIMIC-learned)
      m_yj  <- bin_metrics_at(prob, yte, tau_yj[[cls]], cls, d)
      
      if (site=="ICHT") {
        primary_bin_icht_t05  <- bind_rows(primary_bin_icht_t05,  m_t05)
        primary_bin_icht_spec <- bind_rows(primary_bin_icht_spec, m_spc)
        primary_bin_icht_yj   <- bind_rows(primary_bin_icht_yj,   m_yj)
      } else {
        primary_bin_aumc_t05  <- bind_rows(primary_bin_aumc_t05,  m_t05)
        primary_bin_aumc_spec <- bind_rows(primary_bin_aumc_spec, m_spc)
        primary_bin_aumc_yj   <- bind_rows(primary_bin_aumc_yj,   m_yj)
      }
    }
  }
  
  # ===== RECAL (ICHT & AUMC; metrics @ 0.5; fresh calibration) =====
  # ICHT
  param_mc_i <- c(param_mc, list(num_class = length(levels(tr_i$class))))
  dmat_i_mc  <- xgb.DMatrix(as.matrix(tr_i[, preds]), label = as.numeric(tr_i$class) - 1)
  cv_i_mc    <- safe_xgb_cv(param_mc_i, dmat_i_mc, 300, 5, early_stopping_rounds = 20, verbose = 0, label = paste0("Day ", d, " ICHT MC"))
  mc_i       <- xgb.train(param_mc_i, dmat_i_mc, nrounds = safe_best_iter(cv_i_mc, default = 50), verbose = 0)
  xgb.save(mc_i, file.path(dir_mod_icht_mc, sprintf("day%d_multiclass.json", d)))

  te_i2 <- te_i[!is.na(te_i$class), , drop = FALSE]
  if (nrow(te_i2)>0L) {
    # multiclass
    pmat_recal_i <- predict(mc_i, xgb.DMatrix(as.matrix(te_i2[, preds]))) %>% matrix(ncol = length(class_levels), byrow = TRUE)
    Y_onehot_ri  <- onehot_levels(te_i2$class, class_levels)
    cm_mc_ri     <- confusionMatrix(factor(class_levels[max.col(pmat_recal_i)], levels = class_levels), te_i2$class)
    recal_mc_icht <- bind_rows(recal_mc_icht, tibble(day = d, Accuracy = cm_mc_ri$overall["Accuracy"],
                                                     Brier = mean((pmat_recal_i - Y_onehot_ri)^2),
                                                     ECE   = ece_brier(as.numeric(pmat_recal_i), as.numeric(Y_onehot_ri))$ece))
    # binary @ 0.5 + calibration (using tuned hyperparameters)
    for (cls in class_levels) {
      ytr <- as.numeric(tr_i$class == cls)
      if (length(unique(ytr)) < 2) next
      spw <- sum(ytr==0)/sum(ytr==1)
      # Use class-specific tuned parameters if available
      base_params <- if (!is.null(param_bin_tuned[[cls]])) param_bin_tuned[[cls]] else param_bin
      param_bin_w <- modifyList(base_params, list(scale_pos_weight = spw))
      dmat <- xgb.DMatrix(as.matrix(tr_i[, preds]), label = ytr)
      cvb  <- safe_xgb_cv(param_bin_w, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0, label = paste0("Day ", d, " ICHT ", cls))
      bst  <- xgb.train(param_bin_w, dmat, nrounds = safe_best_iter(cvb, default = 50), verbose = 0)
      xgb.save(bst, file.path(dir_mod_icht_bin, sprintf("day%d_%s.json", d, cls)))

      prob <- predict(bst, xgb.DMatrix(as.matrix(te_i2[, preds])))
      yte  <- as.numeric(te_i2$class == cls)

      # T05 threshold
      recal_bin_icht   <- bind_rows(recal_bin_icht, bin_metrics_at(prob, yte, 0.5, cls, d))
      cal_icht_recal_bin <- bind_rows(cal_icht_recal_bin, collect_cal_bins(prob, yte, cls, d, "ICHT_recal"))
      
      # SPEC threshold (learned from ICHT training data)
      prob_tr_i <- predict(bst, xgb.DMatrix(as.matrix(tr_i[, preds])))
      ytr_eval <- as.numeric(tr_i$class == cls)
      tau_spec_i <- threshold_for_specificity(ytr_eval, prob_tr_i, spec_target = spec_targets[[cls]])
      recal_bin_icht_spec <- bind_rows(recal_bin_icht_spec, bin_metrics_at(prob, yte, tau_spec_i, cls, d))
    }
  }
  
  # AUMC
  param_mc_a <- c(param_mc, list(num_class = length(levels(tr_a$class))))
  dmat_a_mc  <- xgb.DMatrix(as.matrix(tr_a[, preds]), label = as.numeric(tr_a$class) - 1)
  cv_a_mc    <- safe_xgb_cv(param_mc_a, dmat_a_mc, 300, 5, early_stopping_rounds = 20, verbose = 0, label = paste0("Day ", d, " AUMC MC"))
  mc_a       <- xgb.train(param_mc_a, dmat_a_mc, nrounds = safe_best_iter(cv_a_mc, default = 50), verbose = 0)
  xgb.save(mc_a, file.path(dir_mod_aumc_mc, sprintf("day%d_multiclass.json", d)))

  te_a2 <- te_a[!is.na(te_a$class), , drop = FALSE]
  if (nrow(te_a2)>0L) {
    # multiclass
    pmat_recal_a <- predict(mc_a, xgb.DMatrix(as.matrix(te_a2[, preds]))) %>% matrix(ncol = length(class_levels), byrow = TRUE)
    Y_onehot_ra  <- onehot_levels(te_a2$class, class_levels)
    cm_mc_ra     <- confusionMatrix(factor(class_levels[max.col(pmat_recal_a)], levels = class_levels), te_a2$class)
    recal_mc_aumc <- bind_rows(recal_mc_aumc, tibble(day = d, Accuracy = cm_mc_ra$overall["Accuracy"],
                                                     Brier = mean((pmat_recal_a - Y_onehot_ra)^2),
                                                     ECE   = ece_brier(as.numeric(pmat_recal_a), as.numeric(Y_onehot_ra))$ece))
    # binary @ 0.5 + calibration (using tuned hyperparameters)
    for (cls in class_levels) {
      ytr <- as.numeric(tr_a$class == cls)
      if (length(unique(ytr)) < 2) next
      spw <- sum(ytr==0)/sum(ytr==1)
      # Use class-specific tuned parameters if available
      base_params <- if (!is.null(param_bin_tuned[[cls]])) param_bin_tuned[[cls]] else param_bin
      param_bin_w <- modifyList(base_params, list(scale_pos_weight = spw))
      dmat <- xgb.DMatrix(as.matrix(tr_a[, preds]), label = ytr)
      cvb  <- safe_xgb_cv(param_bin_w, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0, label = paste0("Day ", d, " AUMC ", cls))
      bst  <- xgb.train(param_bin_w, dmat, nrounds = safe_best_iter(cvb, default = 50), verbose = 0)
      xgb.save(bst, file.path(dir_mod_aumc_bin, sprintf("day%d_%s.json", d, cls)))

      prob <- predict(bst, xgb.DMatrix(as.matrix(te_a2[, preds])))
      yte  <- as.numeric(te_a2$class == cls)

      # T05 threshold
      recal_bin_aumc   <- bind_rows(recal_bin_aumc, bin_metrics_at(prob, yte, 0.5, cls, d))
      cal_aumc_recal_bin <- bind_rows(cal_aumc_recal_bin, collect_cal_bins(prob, yte, cls, d, "AUMC_recal"))
      
      # SPEC threshold (learned from AUMC training data)
      prob_tr_a <- predict(bst, xgb.DMatrix(as.matrix(tr_a[, preds])))
      ytr_eval <- as.numeric(tr_a$class == cls)
      tau_spec_a <- threshold_for_specificity(ytr_eval, prob_tr_a, spec_target = spec_targets[[cls]])
      recal_bin_aumc_spec <- bind_rows(recal_bin_aumc_spec, bin_metrics_at(prob, yte, tau_spec_a, cls, d))
    }
  }
}


# ---------------- 5b. Feature Importance Analysis --------------------------
message("\n==== FEATURE IMPORTANCE ANALYSIS ====\n")

# Save feature importance tables
dir_fi <- file.path(root_dir, "Feature_importance")
dir.create(dir_fi, recursive = TRUE, showWarnings = FALSE)

write_csv(fi_multiclass, file.path(dir_fi, "feature_importance_multiclass.csv"))
write_csv(fi_binary,     file.path(dir_fi, "feature_importance_binary.csv"))

# Create summary tables for Day 3 (the tuned day)
day3_mc_fi <- fi_multiclass %>% 
  filter(day == 3) %>%
  arrange(desc(Gain)) %>%
  select(Feature, Gain, Cover, Frequency)

day3_bin_fi <- fi_binary %>%
  filter(day == 3) %>%
  group_by(class) %>%
  arrange(desc(Gain), .by_group = TRUE) %>%
  select(class, Feature, Gain, Cover, Frequency)

cat("\n--- Day 3 Multiclass Feature Importance (Top 12) ---\n")
print(day3_mc_fi, n = 12)

cat("\n--- Day 3 Binary Feature Importance by Class (Top 5 each) ---\n")
day3_bin_fi %>% slice_head(n = 5) %>% print(n = 20)

# Create feature importance heatmap for Day 3
fi_day3_wide <- fi_binary %>%
  filter(day == 3) %>%
  select(class, Feature, Gain) %>%
  tidyr::pivot_wider(names_from = class, values_from = Gain, values_fill = 0)

# Add multiclass importance
mc_day3 <- fi_multiclass %>% filter(day == 3) %>% select(Feature, Gain) %>% rename(Multiclass = Gain)
fi_day3_combined <- left_join(fi_day3_wide, mc_day3, by = "Feature") %>%
  mutate(across(where(is.numeric), ~replace_na(., 0)))

cat("\n--- Day 3 Feature Importance Heatmap Data ---\n")
print(fi_day3_combined, n = 50)

write_csv(fi_day3_combined, file.path(dir_fi, "feature_importance_day3_heatmap.csv"))

# Plot feature importance for Day 3
if (nrow(fi_multiclass) > 0) {
  p_fi_mc <- fi_multiclass %>%
    filter(day == 3) %>%
    slice_head(n = 12) %>%
    mutate(Feature = factor(Feature, levels = rev(Feature))) %>%
    ggplot(aes(x = Gain, y = Feature)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    labs(title = "Day 3 Multiclass: Feature Importance (Gain)",
         x = "Gain", y = NULL) +
    theme_minimal(base_size = 14)
  
  ggsave(file.path(dir_fi, "day3_multiclass_feature_importance.png"), p_fi_mc, 
         width = 8, height = 6, dpi = 150)
}

if (nrow(fi_binary) > 0) {
  p_fi_bin <- fi_binary %>%
    filter(day == 3) %>%
    group_by(class) %>%
    slice_head(n = 8) %>%
    ungroup() %>%
    mutate(Feature = reorder(Feature, Gain)) %>%
    ggplot(aes(x = Gain, y = Feature, fill = class)) +
    geom_col(alpha = 0.8) +
    facet_wrap(~class, scales = "free_y", ncol = 2) +
    scale_fill_manual(values = c("C1" = "deeppink", "C2" = "deepskyblue", 
                                  "C3" = "forestgreen", "C4" = "orangered")) +
    labs(title = "Day 3 Binary Classifiers: Feature Importance (Gain)",
         x = "Gain", y = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
  
  ggsave(file.path(dir_fi, "day3_binary_feature_importance.png"), p_fi_bin, 
         width = 10, height = 8, dpi = 150)
}

message("[Feature Importance] Saved to: ", dir_fi)


# ---------------- 6.  write CSV + quick plots (font size = 18) -------------
# Primary (frozen) binary: MIMIC (holdout)
write_csv(primary_bin_mimic_t05,  file.path(dir_primary_bin,"frozen_binary_byDay_MIMIC_thr05.csv"))
write_csv(primary_bin_mimic_spec, file.path(dir_primary_bin,"frozen_binary_byDay_MIMIC_spec.csv"))
write_csv(primary_bin_mimic_yj,   file.path(dir_primary_bin,"frozen_binary_byDay_MIMIC_youden.csv"))

# Primary (frozen) binary: ICHT/AUMC using MIMIC model & thresholds
write_csv(primary_bin_icht_t05,  file.path(dir_primary_bin,"frozen_binary_byDay_ICHT_thr05.csv"))
write_csv(primary_bin_icht_spec, file.path(dir_primary_bin,"frozen_binary_byDay_ICHT_spec.csv"))
write_csv(primary_bin_icht_yj,   file.path(dir_primary_bin,"frozen_binary_byDay_ICHT_youden.csv"))

write_csv(primary_bin_aumc_t05,  file.path(dir_primary_bin,"frozen_binary_byDay_AUMC_thr05.csv"))
write_csv(primary_bin_aumc_spec, file.path(dir_primary_bin,"frozen_binary_byDay_AUMC_spec.csv"))
write_csv(primary_bin_aumc_yj,   file.path(dir_primary_bin,"frozen_binary_byDay_AUMC_youden.csv"))

# Recal (site trained) binary at 0.5 and SPEC
write_csv(recal_bin_icht, file.path(dir_recal_bin,"recal_binary_byDay_ICHT_thr05.csv"))
write_csv(recal_bin_aumc, file.path(dir_recal_bin,"recal_binary_byDay_AUMC_thr05.csv"))
write_csv(recal_bin_icht_spec, file.path(dir_recal_bin,"recal_binary_byDay_ICHT_spec.csv"))
write_csv(recal_bin_aumc_spec, file.path(dir_recal_bin,"recal_binary_byDay_AUMC_spec.csv"))

# Multiclass refs
write_csv(frozen_mc_mimic, file.path(dir_primary_mc,"frozen_multiclass_byDay_MIMIC_holdout.csv"))
write_csv(primary_mc_icht, file.path(dir_primary_mc,"frozen_multiclass_byDay_ICHT.csv"))
write_csv(primary_mc_aumc, file.path(dir_primary_mc,"frozen_multiclass_byDay_AUMC.csv"))
write_csv(recal_mc_icht,   file.path(dir_recal_mc,"recal_multiclass_byDay_ICHT.csv"))
write_csv(recal_mc_aumc,   file.path(dir_recal_mc,"recal_multiclass_byDay_AUMC.csv"))

library(patchwork)
suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(patchwork); library(cowplot)
})


safe_is_nonempty <- function(df) is.data.frame(df) && nrow(df) > 0L

class_levels     <- paste0("C", 1:4)
class_cols_named <- c("C1" = "deeppink",
                      "C2" = "deepskyblue",
                      "C3" = "forestgreen",
                      "C4" = "orangered")

as_class_factor <- function(df) {
  df %>% dplyr::mutate(class = factor(class, levels = class_levels))
}

# If you don't already have it:
# theme_font18 <- theme_bw(base_size = 18)
plot_binary_panel <- function(df, title_stub, out_fp, legend_fp = NULL){
  if (!safe_is_nonempty(df)) {
    message("[plot_binary_panel] Nothing to plot for: ", title_stub)
    return(invisible(NULL))
  }
  req_cols <- c("day","class","AUC","Accuracy","Specificity","Sensitivity","Brier","ECE")
  miss <- setdiff(req_cols, names(df))
  if (length(miss)) stop("Data frame missing required columns: ", paste(miss, collapse=", "))
  
  df <- as_class_factor(df)
  metrics <- c("AUC","Accuracy","Specificity","Sensitivity","Brier","ECE")
  
  mk <- function(m) {
    yl <- if (m %in% c("AUC","Accuracy","Specificity","Sensitivity")) c(0,1) else NULL
    p  <- ggplot(df, aes(x = day, y = .data[[m]], colour = class, group = class)) +
      geom_line(linewidth = 1) +
      scale_color_manual(values = class_cols_named, drop = FALSE, name = "Class") +
      labs(title = m, x = "Last day included (0–d)", y = m) +
      theme_font18 + theme(legend.position = "none")
    if (!is.null(yl)) p <- p + scale_y_continuous(limits = yl, oob = scales::squish)
    p
  }
  
  plist <- lapply(metrics, mk)
  panel <- patchwork::wrap_plots(plist, ncol = 3)
  
  # Title (no & operator)
  panel_with_title <- cowplot::ggdraw() +
    cowplot::draw_label(title_stub, fontface = "bold", size = 18,
                        x = 0.5, y = 0.98, hjust = 0.5, vjust = 1) +
    cowplot::draw_plot(panel, x = 0, y = 0, width = 1, height = 0.95)
  
  ggsave(out_fp, panel_with_title, width = 16, height = 9, dpi = 300)
  message("Saved panel: ", out_fp)
  
  # ---------- clean legend (no warnings) ----------
  if (is.null(legend_fp)) {
    legend_fp <- sub("\\.png$", "_legend.png", out_fp, ignore.case = TRUE)
  }
  # minimal data with one row per class; use points so only the colour guide is built
  legend_df <- df %>% distinct(class) %>% mutate(dummy_x = 0, dummy_y = 0.5)
  p_legend <- ggplot(legend_df, aes(dummy_x, dummy_y, colour = class)) +
    geom_point(size = 4) +
    scale_color_manual(values = class_cols_named, drop = FALSE, name = "Class") +
    guides(color = guide_legend(title = "Class", override.aes = list(linetype = 0, size = 5))) +
    theme_bw(base_size = 18) +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      axis.title = element_blank(),
      axis.text  = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank()
    )
  
  # extract and save (wrap in suppressWarnings just in case)
  leg <- suppressWarnings(cowplot::get_legend(p_legend))
  ggsave(legend_fp, cowplot::ggdraw(leg), width = 6.5, height = 1.2, dpi = 300)
  message("Saved legend: ", legend_fp)
  
  invisible(list(panel = out_fp, legend = legend_fp))
}


# Build panels
plot_binary_panel(primary_bin_mimic_t05,  "MIMIC (thr=0.5)",  file.path(dir_primary_bin, "PANEL_MIMIC_thr05.png"))
plot_binary_panel(primary_bin_mimic_spec, "MIMIC (SPEC)",     file.path(dir_primary_bin, "PANEL_MIMIC_spec.png"))
plot_binary_panel(primary_bin_mimic_yj,   "MIMIC (Youden-J)", file.path(dir_primary_bin, "PANEL_MIMIC_youden.png"))

plot_binary_panel(primary_bin_icht_spec, "ICHT frozen (SPEC)", file.path(dir_primary_bin, "PANEL_ICHT_frozen_SPEC.png"))
plot_binary_panel(primary_bin_icht_t05,  "ICHT frozen (0.5)",  file.path(dir_primary_bin, "PANEL_ICHT_frozen_thr05.png"))
plot_binary_panel(primary_bin_icht_yj,   "ICHT frozen (YJ)",   file.path(dir_primary_bin, "PANEL_ICHT_frozen_YJ.png"))

plot_binary_panel(primary_bin_aumc_spec, "AUMC frozen (SPEC)", file.path(dir_primary_bin, "PANEL_AUMC_frozen_SPEC.png"))
plot_binary_panel(primary_bin_aumc_t05,  "AUMC frozen (0.5)",  file.path(dir_primary_bin, "PANEL_AUMC_frozen_thr05.png"))
plot_binary_panel(primary_bin_aumc_yj,   "AUMC frozen (YJ)",   file.path(dir_primary_bin, "PANEL_AUMC_frozen_YJ.png"))

plot_binary_panel(recal_bin_icht, "ICHT recal (0.5)", file.path(dir_recal_bin, "PANEL_ICHT_recal_thr05.png"))
plot_binary_panel(recal_bin_aumc, "AUMC recal (0.5)", file.path(dir_recal_bin, "PANEL_AUMC_recal_thr05.png"))
plot_binary_panel(recal_bin_icht_spec, "ICHT recal (SPEC)", file.path(dir_recal_bin, "PANEL_ICHT_recal_SPEC.png"))
plot_binary_panel(recal_bin_aumc_spec, "AUMC recal (SPEC)", file.path(dir_recal_bin, "PANEL_AUMC_recal_SPEC.png"))



# One faceted reliability panel per condition store
plot_cal_panel <- function(cal_df, title, out_fp){
  if (nrow(cal_df)==0) return(invisible(NULL))
  cal_df <- cal_df %>%
    mutate(class = factor(class, levels = paste0("C",1:4)),
           day   = factor(day, levels = 0:7, labels = paste0("Day ",0:7)))
  p <- ggplot(cal_df, aes(exp, obs, size = bin_count)) +
    geom_point(alpha = .7) +
    geom_abline(linetype="dashed") +
    coord_equal(xlim=c(0,1), ylim=c(0,1)) +
    scale_size_continuous(range = c(1, 5)) +
    labs(title = title, x = "Predicted", y = "Observed", size = "n") +
    facet_grid(rows = vars(day), cols = vars(class), switch = "y") +
    theme_bw() + theme_font18
  ggsave(out_fp, p, width = 16, height = 22, dpi = 300)
}

plot_cal_panel(cal_mimic_bin,       "Calibration – MIMIC binary (holdout)", file.path(dir_primary_bin,"CAL_PANEL_MIMIC.png"))
plot_cal_panel(cal_icht_frozen_bin, "Calibration – ICHT binary (frozen)",   file.path(dir_primary_bin,"CAL_PANEL_ICHT_frozen.png"))
plot_cal_panel(cal_icht_recal_bin,  "Calibration – ICHT binary (recal)",    file.path(dir_recal_bin,  "CAL_PANEL_ICHT_recal.png"))
plot_cal_panel(cal_aumc_frozen_bin, "Calibration – AUMC binary (frozen)",   file.path(dir_primary_bin,"CAL_PANEL_AUMC_frozen.png"))
plot_cal_panel(cal_aumc_recal_bin,  "Calibration – AUMC binary (recal)",    file.path(dir_recal_bin,  "CAL_PANEL_AUMC_recal.png"))

# Build a compact compare table for ICHT frozen at three thresholds (example)
combine_with_tag <- function(df, tag){
  df %>% mutate(tag = tag) %>%
    select(tag, day, class, AUC, Accuracy, Sensitivity, Specificity, Brier, ECE)
}
icht_compare <- bind_rows(
  combine_with_tag(primary_bin_icht_spec, "SPEC"),
  combine_with_tag(primary_bin_icht_t05,  "T05"),
  combine_with_tag(primary_bin_icht_yj,   "YJ")
) %>% arrange(day, class, tag)

aumc_compare <- bind_rows(
  combine_with_tag(primary_bin_aumc_spec, "SPEC"),
  combine_with_tag(primary_bin_aumc_t05,  "T05"),
  combine_with_tag(primary_bin_aumc_yj,   "YJ")
) %>% arrange(day, class, tag)

mimic_compare <- bind_rows(
  combine_with_tag(primary_bin_mimic_spec, "SPEC"),
  combine_with_tag(primary_bin_mimic_t05,  "T05"),
  combine_with_tag(primary_bin_mimic_yj,   "YJ")
) %>% arrange(day, class, tag)

out_dir_tables <- file.path(root_dir, "tables"); dir.create(out_dir_tables, showWarnings = FALSE, recursive = TRUE)
write_csv(icht_compare,  file.path(out_dir_tables, "compare_ICHT_frozen_SPEC_T05_YJ.csv"))
write_csv(aumc_compare,  file.path(out_dir_tables, "compare_AUMC_frozen_SPEC_T05_YJ.csv"))
write_csv(mimic_compare, file.path(out_dir_tables, "compare_MIMIC_holdout_SPEC_T05_YJ.csv"))

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(stringr)
  library(gt)
})

# ---- file paths produced by your loop (thr = 0.5) ----
f_mimic_fp <- file.path(dir_primary_bin, "frozen_binary_byDay_MIMIC_thr05.csv")
f_icht_fp  <- file.path(dir_primary_bin, "frozen_binary_byDay_ICHT_thr05.csv")
f_aumc_fp  <- file.path(dir_primary_bin, "frozen_binary_byDay_AUMC_thr05.csv")

r_icht_fp  <- file.path(dir_recal_bin,  "recal_binary_byDay_ICHT_thr05.csv")
r_aumc_fp  <- file.path(dir_recal_bin,  "recal_binary_byDay_AUMC_thr05.csv")


safe_read <- function(fp){
  if (!file.exists(fp)) stop("Missing file: ", fp)
  read_csv(fp, show_col_types = FALSE) %>%
    mutate(class = as.character(class))
}

# ---------- 1) Build tidy LONG frames ----------
to_long <- function(df, site, tag){
  df %>%
    select(day, class, AUC, Accuracy, Sensitivity, Specificity, Brier, ECE) %>%
    pivot_longer(cols = -c(day, class), names_to = "metric", values_to = "value") %>%
    mutate(site = site, tag = tag)
}

f_long <- bind_rows(
  to_long(safe_read(f_mimic_fp), "MIMIC", "F"),
  to_long(safe_read(f_icht_fp) , "ICHT",  "F"),
  to_long(safe_read(f_aumc_fp) , "AUMC",  "F")
)

r_long <- bind_rows(
  to_long(safe_read(r_icht_fp), "ICHT", "R"),
  to_long(safe_read(r_aumc_fp), "AUMC", "R")
)

# ---------- 2) Pivot WIDE by site (one column per site per metric) ----------
pivot_by_site <- function(long_df){
  # Ensure metric order and class order are stable
  long_df <- long_df %>%
    mutate(
      metric = factor(metric, levels = c("AUC","Accuracy","Sensitivity","Specificity","Brier","ECE")),
      class  = factor(class,  levels = paste0("C",1:4), ordered = TRUE)
    )
  
  wide <- long_df %>%
    unite(col = "colname", site, metric, sep = " • ", remove = FALSE) %>%
    select(day, class, colname, value) %>%
    pivot_wider(names_from = colname, values_from = value) %>%
    arrange(day, class)
  
  # Order columns: day, class, then per-site spanners in metric order
  sites_present   <- c("MIMIC","ICHT","AUMC")
  metrics_present <- c("AUC","Accuracy","Sensitivity","Specificity","Brier","ECE")
  desired_cols <- c("day","class",
                    as.vector(sapply(sites_present, function(s) paste0(s, " • ", metrics_present))))
  desired_cols <- desired_cols[desired_cols %in% names(wide)]
  wide[, desired_cols, drop = FALSE]
}

tbl_frozen_wide <- pivot_by_site(f_long)
tbl_recal_wide  <- pivot_by_site(r_long)

# ---------- 3) Rounding ----------
round_wide <- function(df){
  num_cols <- setdiff(names(df), c("day","class"))
  brier_ece <- grep("Brier|ECE", num_cols, value = TRUE)
  other     <- setdiff(num_cols, brier_ece)
  df %>%
    mutate(
      across(all_of(other),     ~ round(., 2)),
      across(all_of(brier_ece), ~ round(., 3))
    )
}

tbl_frozen_out <- round_wide(tbl_frozen_wide)
tbl_recal_out  <- round_wide(tbl_recal_wide)

# ---------- 4) Write CSVs ----------
out_dir_tables <- file.path(root_dir, "tables")
dir.create(out_dir_tables, showWarnings = FALSE, recursive = TRUE)

frozen_csv <- file.path(out_dir_tables, "binary_metrics_FROZEN_MIMIC_ICHT_AUMC_thr05.csv")
recal_csv  <- file.path(out_dir_tables, "binary_metrics_RECAL_ICHT_AUMC_thr05.csv")

write_csv(tbl_frozen_out, frozen_csv)
write_csv(tbl_recal_out,  recal_csv)
message("✅ Tables written:\n  - ", frozen_csv, "\n  - ", recal_csv)

# ---------- 5) GT tables (no re-pivoting) ----------
gt_from_wide <- function(df_wide, title_text, subtitle_text = NULL){
  df_wide <- df_wide %>%
    mutate(class = factor(class, levels = paste0("C",1:4))) %>%
    arrange(day, class)
  
  metric_cols <- setdiff(names(df_wide), c("day","class"))
  site_cols <- list(
    MIMIC = grep("^MIMIC • ", metric_cols, value = TRUE),
    ICHT  = grep("^ICHT • " , metric_cols, value = TRUE),
    AUMC  = grep("^AUMC • " , metric_cols, value = TRUE)
  )
  # Pretty labels (strip 'SITE • ')
  col_labels <- setNames(gsub("^[^•]+ •\\s*", "", unlist(site_cols)), unlist(site_cols))
  
  df_wide %>%
    gt() %>%
    tab_header(title = title_text, subtitle = subtitle_text) %>%
    { if (length(site_cols$MIMIC)) tab_spanner(., "MIMIC", columns = all_of(site_cols$MIMIC)) else . } %>%
    { if (length(site_cols$ICHT))  tab_spanner(., "ICHT",  columns = all_of(site_cols$ICHT))  else . } %>%
    { if (length(site_cols$AUMC))  tab_spanner(., "AUMC",  columns = all_of(site_cols$AUMC))  else . } %>%
    cols_label(.list = col_labels) %>%
    tab_options(
      table.font.size = px(14),
      data_row.padding = px(4),
      heading.title.font.size = px(18),
      column_labels.font.size = px(14)
    )
}

gt_frozen <- gt_from_wide(
  tbl_frozen_out,
  "Binary metrics (threshold = 0.5) – FROZEN models",
  "MIMIC = hold-out; ICHT/AUMC = external evaluation with frozen MIMIC models"
)
gt_recal  <- gt_from_wide(
  tbl_recal_out,
  "Binary metrics (threshold = 0.5) – RECALIBRATED models",
  "ICHT/AUMC retrained (80/20 split within site)"
)

gtsave(gt_frozen, file.path(out_dir_tables, "binary_metrics_FROZEN_thr05.html"))
gtsave(gt_recal,  file.path(out_dir_tables, "binary_metrics_RECAL_thr05.html"))

# ========== SPEC THRESHOLD TABLES ==========
# Read SPEC threshold files
f_mimic_spec_fp <- file.path(dir_primary_bin, "frozen_binary_byDay_MIMIC_spec.csv")
f_icht_spec_fp  <- file.path(dir_primary_bin, "frozen_binary_byDay_ICHT_spec.csv")
f_aumc_spec_fp  <- file.path(dir_primary_bin, "frozen_binary_byDay_AUMC_spec.csv")

f_long_spec <- bind_rows(
  to_long(safe_read(f_mimic_spec_fp), "MIMIC", "F"),
  to_long(safe_read(f_icht_spec_fp) , "ICHT",  "F"),
  to_long(safe_read(f_aumc_spec_fp) , "AUMC",  "F")
)

tbl_frozen_wide_spec <- pivot_by_site(f_long_spec)
tbl_frozen_out_spec  <- round_wide(tbl_frozen_wide_spec)

frozen_spec_csv <- file.path(out_dir_tables, "binary_metrics_FROZEN_MIMIC_ICHT_AUMC_spec.csv")
write_csv(tbl_frozen_out_spec, frozen_spec_csv)

gt_frozen_spec <- gt_from_wide(
  tbl_frozen_out_spec,
  "Binary metrics (SPEC threshold) – FROZEN models",
  "MIMIC = hold-out; ICHT/AUMC = external evaluation with frozen MIMIC models"
)
gtsave(gt_frozen_spec, file.path(out_dir_tables, "binary_metrics_FROZEN_spec.html"))
message("SPEC tables written:\n  - ", frozen_spec_csv, "\n  - binary_metrics_FROZEN_spec.html")

# ========== SPEC THRESHOLD TABLES FOR RECAL ==========
# Read SPEC threshold files for recalibrated models
r_icht_spec_fp <- file.path(dir_recal_bin, "recal_binary_byDay_ICHT_spec.csv")
r_aumc_spec_fp <- file.path(dir_recal_bin, "recal_binary_byDay_AUMC_spec.csv")

r_long_spec <- bind_rows(
  to_long(safe_read(r_icht_spec_fp), "ICHT", "R"),
  to_long(safe_read(r_aumc_spec_fp), "AUMC", "R")
)

tbl_recal_wide_spec <- pivot_by_site(r_long_spec)
tbl_recal_out_spec  <- round_wide(tbl_recal_wide_spec)

recal_spec_csv <- file.path(out_dir_tables, "binary_metrics_RECAL_ICHT_AUMC_spec.csv")
write_csv(tbl_recal_out_spec, recal_spec_csv)

gt_recal_spec <- gt_from_wide(
  tbl_recal_out_spec,
  "Binary metrics (SPEC threshold) – RECALIBRATED models",
  "ICHT/AUMC retrained (80/20 split within site); SPEC thresholds learned from training data"
)
gtsave(gt_recal_spec, file.path(out_dir_tables, "binary_metrics_RECAL_spec.html"))
message("RECAL SPEC tables written:\n  - ", recal_spec_csv, "\n  - binary_metrics_RECAL_spec.html")

# ---------- 6) Quick sanity checks ----------
# Different values across classes for same day?
print(
  tbl_frozen_wide %>%
    filter(day %in% c(0,7)) %>%
    select(day, class, `ICHT • AUC`, `AUMC • AUC`, `MIMIC • AUC`) %>%
    arrange(day, class)
)
# ─────────────────────────────────────────────────────────────────────────────
# =========================  DAILY TRAJECTORIES (ICHT vs AUMC)  =========================
# Inputs expected to exist:
#   long_icht, long_aumc, prob_icht, prob_aumc
#   class_levels, class_cols_named, theme_custom (optional)

# ----------------- Config -----------------
row_vars <- c("avg_peep","avg_pco2","norad_vasorate",
              "avg_lactate","locf_creatinine","locf_bicarbonate")

var_labs <- c(
  avg_peep         = "PEEP (cmH\u2082O)",
  avg_pco2         = "PaCO\u2082 (mmHg)",
  norad_vasorate   = "Norepinephrine (\u03bcg/kg/min)",
  avg_lactate      = "Lactate (mmol/L)",
  locf_creatinine  = "Creatinine (mg/dL)",
  locf_bicarbonate = "Bicarbonate (mmol/L)"
)

# sensible defaults if not already defined
if (!exists("class_levels"))     class_levels     <- paste0("C", 1:4)
if (!exists("class_cols_named")) class_cols_named <- c("C1"="deeppink","C2"="deepskyblue","C3"="forestgreen","C4"="orangered")
if (!exists("theme_custom"))     theme_custom     <- theme_minimal(base_size = 14)

dir_traj <- file.path(getwd(), "Trajectory panels")
dir.create(dir_traj, showWarnings = FALSE, recursive = TRUE)
prof_out   <- file.path(dir_traj, "var_profiles_faceted")
dir.create(prof_out, showWarnings = FALSE, recursive = TRUE)

panel_file  <- file.path(prof_out, "trajectories_ICHT_vs_AUMC_no_legend.png")
legend_file <- file.path(prof_out, "trajectories_legend.png")

# ----------------- Helpers -----------------
make_var_long <- function(long_df, prob_df) {
  long_df %>%
    dplyr::filter(days_from_start %in% 0:14) %>%
    dplyr::select(stay_id, day_period = days_from_start, dplyr::all_of(row_vars)) %>%
    tidyr::pivot_longer(-c(stay_id, day_period), names_to = "variable", values_to = "value") %>%
    dplyr::left_join(
      prob_df %>% dplyr::mutate(class = factor(paste0("C", class), levels = class_levels)),
      by = "stay_id"
    )
}

summarise_site <- function(df_long, site_tag){
  df_long %>%
    dplyr::filter(variable %in% row_vars) %>%
    dplyr::group_by(variable, class, day_period) %>%
    dplyr::summarise(
      n_total = dplyr::n(),
      n_miss  = sum(is.na(value)),
      n_keep  = n_total - n_miss,
      p_miss  = n_miss / n_total,
      mean_v  = mean(value, na.rm = TRUE),
      se_v    = stats::sd(value,  na.rm = TRUE) / sqrt(pmax(n_keep, 1)),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      mean_v = dplyr::if_else(n_keep < 10, NA_real_, mean_v),
      se_v   = dplyr::if_else(n_keep < 10, NA_real_, se_v),
      p_miss = dplyr::if_else(n_keep < 10, NA_real_, p_miss),
      site   = site_tag
    ) %>%
    dplyr::filter(!is.na(mean_v))
}

# ----------------- Build long & summary -----------------
var_long_icht <- make_var_long(long_icht, prob_icht)
var_long_aumc <- make_var_long(long_aumc, prob_aumc)

tab_icht <- summarise_site(var_long_icht, "ICHT")
tab_aumc <- summarise_site(var_long_aumc, "AUMC")

tab_all <- dplyr::bind_rows(tab_icht, tab_aumc) %>%
  dplyr::mutate(
    variable = factor(variable, levels = row_vars, labels = unname(var_labs[row_vars])),
    site     = factor(site, levels = c("ICHT","AUMC")),
    class    = factor(class, levels = class_levels)
  )

# ----------------- Theme -----------------
theme_traj <- theme_custom %+replace% theme(
  legend.position = "none",
  plot.title      = element_blank(),
  axis.title.x    = element_text(size = 16),
  axis.title.y    = element_text(size = 16),
  axis.text       = element_text(size = 13),
  strip.text.y    = element_text(size = 16, face = "bold"),
  strip.text.x    = element_text(size = 15),
  panel.border    = element_rect(colour = "grey30", fill = NA, linewidth = 0.7) # <-- border
)
# ----------------- Base plot (NO legend, NO titles) -----------------
p_base <- ggplot(
  tab_all,
  aes(x = day_period, y = mean_v, colour = class, group = class)
) +
  geom_ribbon(aes(ymin = mean_v - se_v,
                  ymax = mean_v + se_v,
                  fill = class),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(aes(size = p_miss), alpha = 0.7) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, guide = "none", drop = FALSE) +
  scale_size_continuous(
    range  = c(0.5, 4),
    breaks = c(0, 0.25, 0.5, 0.75),
    labels = scales::percent_format(accuracy = 1),
    name   = "% missing"
  ) +
  scale_x_continuous(breaks = seq(0, 14, 2), limits = c(0, 14)) +
  labs(x = "ICU day", y = "Mean \u00B1 SE", colour = "Class") +
  facet_grid(variable ~ site, scales = "free_y") +
  theme_traj

# ----------------- Save main panel -----------------
ggsave(panel_file, p_base, width = 11, height = 14, dpi = 300)
message("Saved trajectory panels (no legend): ", panel_file)

# ----------------- Make and save a separate legend (once) -----------------
# We create a tiny dummy plot that uses the same mappings and palette, then extract its legend.
p_for_legend <- ggplot(
  tab_all |>
    dplyr::distinct(class) |>
    dplyr::mutate(x = 0, y = 0),  # single “point” per class
  aes(x = x, y = y, colour = class)
) +
  geom_point(size = 5) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  guides(colour = guide_legend(title = "Class")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title    = element_text(size = 14, face = "bold"),
    legend.text     = element_text(size = 13)
  )

legend_g <- cowplot::get_legend(p_for_legend)
ggsave(legend_file, plot = cowplot::ggdraw(legend_g), width = 6, height = 1.2, dpi = 300)
message("✅ Saved legend: ", legend_file)



# ---------------------------------------------------------------------
# 7) Making standalone legends
# ---------------------------------------------------------------------

# ---- Palette you already use ----
class_levels     <- paste0("C", 1:4)
class_cols_named <- c("C1" = "deeppink",
                      "C2" = "deepskyblue",
                      "C3" = "forestgreen",
                      "C4" = "orangered")

# ---- Standalone legend maker (no ggplot guides) ----
make_class_legend <- function(out_fp,
                              orientation = c("horizontal","vertical"),
                              labels = names(class_cols_named),
                              colors = unname(class_cols_named),
                              title = "Class",
                              base_size = 18) {
  orientation <- match.arg(orientation)
  stopifnot(length(labels) == length(colors))
  
  df <- data.frame(label = labels, col = colors, stringsAsFactors = FALSE)
  
  if (orientation == "horizontal") {
    df$x <- seq_along(labels)
    df$y <- 1
    p <- ggplot(df, aes(x, y)) +
      # coloured squares
      geom_tile(aes(fill = label), width = 0.8, height = 0.8) +
      # text under each square
      geom_text(aes(label = label), nudge_y = -0.65, size = base_size/3.5) +
      # title above, centered across the strip
      annotate("text", x = mean(range(df$x)), y = 1.7, label = title,
               fontface = "bold", size = base_size/3) +
      scale_fill_manual(values = setNames(df$col, df$label), guide = "none") +
      coord_cartesian(xlim = c(0.5, length(labels) + 0.5), ylim = c(-0.2, 2)) +
      theme_void()
    
    # nice wide aspect
    ggsave(out_fp, p, width = max(6, 1.5 * length(labels)), height = 2, dpi = 300)
  } else {
    df$x <- 1
    # top-to-bottom in class order
    df$y <- rev(seq_along(labels))
    p <- ggplot(df, aes(x, y)) +
      # title on the left
      annotate("text", x = 0.2, y = max(df$y) + 0.4, label = title,
               hjust = 0, fontface = "bold", size = base_size/3) +
      # coloured squares
      geom_tile(aes(fill = label), width = 0.8, height = 0.8) +
      # text to the right of each square
      geom_text(aes(label = label), nudge_x = 0.9, hjust = 0, size = base_size/3.5) +
      scale_fill_manual(values = setNames(df$col, df$label), guide = "none") +
      coord_cartesian(xlim = c(-0.1, 3), ylim = c(0.5, length(labels) + 0.8)) +
      theme_void()
    
    # portrait aspect
    ggsave(out_fp, p, width = 3.6, height = max(2.2, 0.55 * length(labels)), dpi = 300)
  }
  
  message("Saved legend: ", out_fp)
}

# ---- Examples ----
make_class_legend(file.path(dir_primary_bin, "legend_classes_horizontal.png"),
                  orientation = "horizontal",
                  labels = names(class_cols_named),            # "C1"..."C4"
                  colors = unname(class_cols_named),
                  title = "Class")

# If you prefer “Class 1”, “Class 2”, etc (but still colour-mapped the same):
make_class_legend(file.path(dir_primary_bin, "legend_classes_horizontal_words.png"),
                  orientation = "horizontal",
                  labels = paste("Class", 1:4),
                  colors = unname(class_cols_named),
                  title = "Class")

# Vertical version (good for sidebars)
make_class_legend(file.path(dir_primary_bin, "legend_classes_vertical.png"),
                  orientation = "vertical",
                  labels = paste("Class", 1:4),
                  colors = unname(class_cols_named))
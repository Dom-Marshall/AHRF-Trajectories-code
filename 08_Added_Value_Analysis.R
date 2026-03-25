# =============================================================================
# MIMIC: Added-value of predicted trajectory class at Day 3 (leakage-reduced)
# - 80/20 split for MORTALITY evaluation
# - Class-prediction model trained ONLY on TRAIN using day0-3 physiology
# - Compare multiclass vs OvR class predictor (soft + hard)
# - Mortality models:
#   PF only
#   Baseline (PF+SOFA)
#   + MC hard / + MC probs
#   + OvR hard / + OvR probs
#   Plus "class-only" versions (no PF/SOFA) for interest
# - Outputs: AUC+CI, DeLong vs baseline & vs PF, calibration intercept/slope, Brier, DCA net benefit
# =============================================================================

setwd("D:/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories")

# ---------------- Packages ---------------------------------------------------
required_packages <- c(
  "dplyr","tidyr","readr",
  "pROC","ggplot2","scales",
  "xgboost","caret",
  "dcurves"
)

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cran.rstudio.com/")
}

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(pROC); library(ggplot2); library(scales)
  library(xgboost); library(caret)
  library(dcurves)
})

# ---------------- Output directory ------------------------------------------
root_dir <- "Prognostic_Added_Value_v4_tuned_80_20"
dir.create(root_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------- Helpers ----------------------------------------------------
pick_col <- function(df, patterns, required = TRUE) {
  nms <- names(df)
  for (p in patterns) {
    hit <- grep(p, nms, ignore.case = TRUE, value = TRUE)
    if (length(hit) > 0) return(hit[1])
  }
  if (required) stop("Could not find a matching column in: ", paste(nms, collapse = ", "))
  return(NA_character_)
}

clamp_prob <- function(p, eps = 1e-6) pmax(pmin(p, 1 - eps), eps)

# Helper to get best iteration from xgb.cv (compatible with xgboost 3.x)
get_best_iter <- function(cv_result) {
  # xgboost 3.x stores best_iteration inside early_stop
  if (!is.null(cv_result$early_stop$best_iteration)) {
    return(cv_result$early_stop$best_iteration)
  }
  # Older xgboost stored it directly
  if (!is.null(cv_result$best_iteration)) {
    return(cv_result$best_iteration)
  }
  # Fallback to niter if no early stopping occurred
  if (!is.null(cv_result$niter)) {
    return(cv_result$niter)
  }
  stop("Could not determine best iteration from xgb.cv result")
}

calibration_stats <- function(y01, p) {
  p <- clamp_prob(p)
  lp <- qlogis(p)
  
  # Calibration-in-the-large: intercept with slope fixed at 1 via offset
  fit_int <- suppressWarnings(glm(y01 ~ 1, family = binomial(), offset = lp))
  cal_int <- as.numeric(coef(fit_int)[1])
  
  # Calibration slope: logistic regression of outcome on linear predictor
  fit_slope <- suppressWarnings(glm(y01 ~ lp, family = binomial()))
  cal_slope <- as.numeric(coef(fit_slope)[2])
  
  brier <- mean((p - y01)^2)
  
  tibble(cal_intercept = cal_int, cal_slope = cal_slope, brier = brier)
}

auc_ci <- function(y01, p) {
  r <- pROC::roc(y01, p, quiet = TRUE)
  ci <- pROC::ci.auc(r)
  list(roc = r, auc = as.numeric(ci[2]), lo = as.numeric(ci[1]), hi = as.numeric(ci[3]))
}

fmt_auc <- function(x) sprintf("%.3f (%.3f-%.3f)", x$auc, x$lo, x$hi)

delong_p <- function(roc1, roc2) as.numeric(pROC::roc.test(roc1, roc2, method = "delong")$p.value)

# Decision curve wrapper that saves plot + CSV of net benefit curves
save_dca <- function(df, outcome_col, pred_cols, out_prefix,
                     thresholds = seq(0.01, 0.60, by = 0.01)) {
  
  dca_df <- df %>%
    select(all_of(c(outcome_col, pred_cols))) %>%
    mutate(!!outcome_col := as.integer(.data[[outcome_col]]))
  
  form <- as.formula(
    paste0(outcome_col, " ~ ", paste(pred_cols, collapse = " + "))
  )
  
  dca_res <- dcurves::dca(
    formula = form,
    data = dca_df,
    thresholds = thresholds
  )
  
  # Plot
  p <- plot(dca_res, smooth = TRUE, show_ggplot_code = FALSE) +
    theme_minimal(base_size = 14) +
    labs(
      title = "Decision Curve Analysis (TEST)",
      x = "Threshold probability",
      y = "Net benefit"
    )
  
  ggsave(file.path(root_dir, paste0(out_prefix, "_DCA.png")),
         p, width = 10, height = 7, dpi = 300)
  
  # Save underlying net benefit table
  # dcurves objects can be tidied via as_tibble()
  dca_tbl <- dcurves::as_tibble(dca_res)
  write_csv(dca_tbl, file.path(root_dir, paste0(out_prefix, "_DCA_net_benefit.csv")))
  
  invisible(list(dca = dca_res, dca_tbl = dca_tbl))
}

# =============================================================================
# 1) Read data + build Day-3 mortality cohort
# =============================================================================

long_mim <- read_csv("Data/mimic_dynamic_var.csv", show_col_types = FALSE)

# Trajectory labels (precomputed)
prob_mim_raw <- read_csv("Data/pprob_MIMIC.csv", show_col_types = FALSE)

prob_mim <- prob_mim_raw %>%
  {
    if ("Assigned_Trajectory_Class" %in% names(.)) {
      select(., stay_id, Assigned_Trajectory_Class)
    } else if ("class" %in% names(.)) {
      select(., stay_id, class) %>% rename(Assigned_Trajectory_Class = class)
    } else {
      stop("pprob_MIMIC.csv must contain either 'Assigned_Trajectory_Class' or 'class'.")
    }
  } %>%
  mutate(
    Assigned_Trajectory_Class = factor(
      paste0("C", as.integer(gsub("^C", "", as.character(Assigned_Trajectory_Class)))),
      levels = paste0("C", 1:4)
    )
  )

# SOFA
sofa_raw <- read_csv("Data/mimic_3dc_sofa.csv", show_col_types = FALSE)
sofa_id_col <- if ("stay_id" %in% names(sofa_raw)) "stay_id" else pick_col(sofa_raw, c("^stay_id$", "stayid", "icu.*stay"))
sofa_score_col <- pick_col(sofa_raw, c("^sofa_24hours$", "^sofa_24h$", "sofa.*24", "sofa_within_24", "sofa.*within.*24", "^sofa$"))

sofa <- sofa_raw %>%
  transmute(
    stay_id  = as.integer(.data[[sofa_id_col]]),
    SOFA_24h = as.numeric(.data[[sofa_score_col]])
  )

# SOFA Day 3
sofa_ts_raw <- read_csv("Data/mimc_ts_sofa.csv", show_col_types = FALSE)
sofa_day3 <- sofa_ts_raw %>%
  filter(day_num == 3) %>%
  select(stay_id, sofa_total_24h) %>%
  rename(SOFA_day3 = sofa_total_24h)

# Day-3 cohort + 14-day mortality
day3_data <- long_mim %>%
  filter(days_from_start == 3) %>%
  select(
    stay_id,
    timezero, icu_outtime, icu_mort, deathtime,
    avg_pao2fio2ratio,
    admission_age, gender
  ) %>%
  left_join(prob_mim, by = "stay_id") %>%
  left_join(sofa,     by = "stay_id") %>%
  left_join(sofa_day3, by = "stay_id") %>%
  mutate(
    timezero    = as.POSIXct(timezero),
    icu_outtime = as.POSIXct(icu_outtime),
    deathtime   = as.POSIXct(deathtime),
    
    time_to_death_days = case_when(
      !is.na(deathtime) ~ as.numeric(difftime(deathtime, timezero, units = "days")),
      icu_mort == 1 & !is.na(icu_outtime) ~ as.numeric(difftime(icu_outtime, timezero, units = "days")),
      TRUE ~ NA_real_
    ),
    time_to_death_days = ifelse(time_to_death_days < 0, NA_real_, time_to_death_days),
    mort_14day = ifelse(!is.na(time_to_death_days) & time_to_death_days <= 14, 1, 0),
    
    PF_ratio_day3 = ifelse(avg_pao2fio2ratio > 470 | avg_pao2fio2ratio < 30, NA_real_, avg_pao2fio2ratio),
    y01 = mort_14day
  ) %>%
  filter(
    !is.na(PF_ratio_day3),
    !is.na(SOFA_24h),
    !is.na(SOFA_day3),
    !is.na(Assigned_Trajectory_Class)
  ) %>%
  mutate(
    mort_14day = factor(mort_14day, levels = c(0, 1)),
    y01 = as.integer(y01)
  )

if (length(unique(day3_data$y01)) < 2) stop("Outcome has <2 classes after filtering.")

cat(sprintf("\nDay3 cohort: N=%d; events=%d (%.1f%%)\n",
            nrow(day3_data), sum(day3_data$y01), 100*mean(day3_data$y01)))

# =============================================================================
# 2) 80/20 split for mortality evaluation (stratified)
# =============================================================================
set.seed(2025)
idx_tr <- caret::createDataPartition(day3_data$mort_14day, p = 0.80, list = FALSE)
train_ids <- day3_data$stay_id[idx_tr]
test_ids  <- day3_data$stay_id[-idx_tr]

day3_tr <- day3_data %>% filter(stay_id %in% train_ids)
day3_te <- day3_data %>% filter(stay_id %in% test_ids)

cat(sprintf("TRAIN: n=%d, events=%d (%.1f%%)\n", nrow(day3_tr), sum(day3_tr$y01), 100*mean(day3_tr$y01)))
cat(sprintf("TEST : n=%d, events=%d (%.1f%%)\n", nrow(day3_te), sum(day3_te$y01), 100*mean(day3_te$y01)))

# =============================================================================
# 3) Build Day0-3 physiology features for CLASS prediction
# =============================================================================
measure <- c(
  "norad_vasorate","avg_lactate","locf_bicarbonate",
  "avg_peak_insp_pressure","avg_peep","avg_pao2fio2ratio",
  "locf_creatinine","avg_pco2","avg_minute_volume","avg_resp_rate"
)
measure <- intersect(measure, names(long_mim))
if (length(measure) == 0) stop("No shared class-prediction measures found in long_mim.")

static_mim <- long_mim %>%
  mutate(age = as.numeric(admission_age), gender_bin = if_else(gender == "M", 1, 0)) %>%
  select(stay_id, age, gender_bin) %>% distinct()

wide_mim_day03 <- function(stay_ids) {
  long_mim %>%
    filter(stay_id %in% stay_ids, days_from_start %in% 0:3) %>%
    select(stay_id, days_from_start, all_of(measure)) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = days_from_start,
      values_from= all_of(measure),
      names_glue = "day{days_from_start}_{.value}"
    ) %>%
    left_join(static_mim, by = "stay_id") %>%
    left_join(prob_mim,   by = "stay_id") %>%
    mutate(Assigned_Trajectory_Class = factor(Assigned_Trajectory_Class, levels = paste0("C",1:4)))
}

wide_tr <- wide_mim_day03(train_ids) %>% inner_join(day3_tr %>% select(stay_id), by="stay_id")
wide_te <- wide_mim_day03(test_ids)  %>% inner_join(day3_te %>% select(stay_id), by="stay_id")

pred_cols <- setdiff(names(wide_tr), c("stay_id","Assigned_Trajectory_Class"))
X_tr <- as.matrix(wide_tr[, pred_cols])
X_te <- as.matrix(wide_te[, pred_cols])

y_tr_class <- wide_tr$Assigned_Trajectory_Class
class_levels <- paste0("C", 1:4)

# =============================================================================
# 4) Class predictors: multiclass + OvR
# =============================================================================

# --- Tuned hyperparameters from Day 3 hyperparameter optimization (10k models) ---

# Multiclass (multi:softprob) - tuned parameters
param_mc <- list(
  booster         = "gbtree",
  objective       = "multi:softprob",
  eval_metric     = "mlogloss",
  eta             = 0.2,
  max_depth       = 8,
  min_child_weight = 2,
  subsample       = 0.6,
  colsample_bytree = 1.0,
  gamma           = 0.5,
  lambda          = 5,
  alpha           = 0
)

# Binary (OvR) base parameters - will be overridden by class-specific tuned params
param_bin <- list(
  booster         = "gbtree",
  objective       = "binary:logistic",
  eval_metric     = "auc",
  eta             = 0.1,
  max_depth       = 6,
  min_child_weight = 1,
  subsample       = 0.8,
  colsample_bytree = 0.8,
  gamma           = 0,
  lambda          = 1,
  alpha           = 0
)

# Class-specific tuned binary parameters (from Day 3 tuning)
param_bin_tuned <- list(
  C1 = list(eta = 0.01, max_depth = 8, min_child_weight = 1,
            subsample = 0.6, colsample_bytree = 1.0, gamma = 2, lambda = 5, alpha = 0),
  C2 = list(eta = 0.1, max_depth = 3, min_child_weight = 1,
            subsample = 0.6, colsample_bytree = 0.6, gamma = 0, lambda = 5, alpha = 0),
  C3 = list(eta = 0.05, max_depth = 3, min_child_weight = 2,
            subsample = 0.85, colsample_bytree = 0.8, gamma = 0.5, lambda = 10, alpha = 0),
  C4 = list(eta = 0.01, max_depth = 5, min_child_weight = 10,
            subsample = 1.0, colsample_bytree = 0.6, gamma = 0, lambda = 1, alpha = 0.5)
)

# ---- Multiclass ----
set.seed(2025)
dmat_mc_tr <- xgb.DMatrix(X_tr, label = as.numeric(y_tr_class) - 1)
cv_mc <- xgb.cv(params = c(param_mc, list(num_class = length(class_levels))),
                data = dmat_mc_tr, nrounds = 400, nfold = 5,
                early_stopping_rounds = 25, verbose = 0)
best_iter_mc <- get_best_iter(cv_mc)
mc_model <- xgb.train(params = c(param_mc, list(num_class = length(class_levels))),
                      data = dmat_mc_tr, nrounds = best_iter_mc, verbose = 0)

pmat_mc_tr <- predict(mc_model, xgb.DMatrix(X_tr)) %>% matrix(ncol=4, byrow=TRUE)
pmat_mc_te <- predict(mc_model, xgb.DMatrix(X_te)) %>% matrix(ncol=4, byrow=TRUE)
colnames(pmat_mc_tr) <- paste0("mc_p_", class_levels)
colnames(pmat_mc_te) <- paste0("mc_p_", class_levels)

hard_mc_tr <- factor(class_levels[max.col(pmat_mc_tr)], levels = class_levels)
hard_mc_te <- factor(class_levels[max.col(pmat_mc_te)], levels = class_levels)

# ---- OvR binaries ----
ovr_models <- setNames(vector("list", length(class_levels)), class_levels)
pmat_ovr_tr <- matrix(NA_real_, nrow=nrow(X_tr), ncol=4); colnames(pmat_ovr_tr) <- paste0("ovr_p_", class_levels)
pmat_ovr_te <- matrix(NA_real_, nrow=nrow(X_te), ncol=4); colnames(pmat_ovr_te) <- paste0("ovr_p_", class_levels)

set.seed(2025)
for (j in seq_along(class_levels)) {
  cls <- class_levels[j]
  y01_cls <- as.integer(y_tr_class == cls)
  if (length(unique(y01_cls)) < 2) {
    warning(sprintf("OvR %s skipped: degenerate in TRAIN split.", cls))
    next
  }
  spw <- sum(y01_cls == 0) / sum(y01_cls == 1)
  dmat_bin <- xgb.DMatrix(X_tr, label = y01_cls)
  
  # Use class-specific tuned parameters
  class_params <- modifyList(param_bin, param_bin_tuned[[cls]])
  class_params[["scale_pos_weight"]] <- spw
  cv_bin <- xgb.cv(params = class_params,
                   data = dmat_bin, nrounds = 400, nfold = 5,
                   early_stopping_rounds = 25, verbose = 0)
  best_iter_bin <- get_best_iter(cv_bin)
  bst <- xgb.train(params = class_params,
                   data = dmat_bin, nrounds = best_iter_bin, verbose = 0)
  ovr_models[[cls]] <- bst
  pmat_ovr_tr[, j] <- predict(bst, xgb.DMatrix(X_tr))
  pmat_ovr_te[, j] <- predict(bst, xgb.DMatrix(X_te))
}

hard_ovr_tr <- factor(class_levels[max.col(pmat_ovr_tr)], levels = class_levels)
hard_ovr_te <- factor(class_levels[max.col(pmat_ovr_te)], levels = class_levels)

# =============================================================================
# 5) Attach predicted class features to TRAIN/TEST mortality data
# =============================================================================
pred_tr <- tibble(stay_id = wide_tr$stay_id, hard_mc = hard_mc_tr, hard_ovr = hard_ovr_tr) %>%
  bind_cols(as_tibble(pmat_mc_tr)) %>%
  bind_cols(as_tibble(pmat_ovr_tr))

pred_te <- tibble(stay_id = wide_te$stay_id, hard_mc = hard_mc_te, hard_ovr = hard_ovr_te) %>%
  bind_cols(as_tibble(pmat_mc_te)) %>%
  bind_cols(as_tibble(pmat_ovr_te))

day3_tr2 <- day3_tr %>% left_join(pred_tr, by="stay_id")
day3_te2 <- day3_te %>% left_join(pred_te, by="stay_id")

# =============================================================================
# 6) Fit mortality models on TRAIN only (full + class-only)
# =============================================================================
m_pf       <- glm(mort_14day ~ PF_ratio_day3, data = day3_tr2, family = binomial())
m_base     <- glm(mort_14day ~ PF_ratio_day3 + SOFA_24h, data = day3_tr2, family = binomial())
m_base_d3  <- glm(mort_14day ~ PF_ratio_day3 + SOFA_day3, data = day3_tr2, family = binomial())

# With PF+SOFA (Baseline)
m_mc_hard  <- glm(mort_14day ~ PF_ratio_day3 + SOFA_24h + hard_mc, data = day3_tr2, family = binomial())
m_mc_soft  <- glm(mort_14day ~ PF_ratio_day3 + SOFA_24h + mc_p_C2 + mc_p_C3 + mc_p_C4, data = day3_tr2, family = binomial())
m_ovr_hard <- glm(mort_14day ~ PF_ratio_day3 + SOFA_24h + hard_ovr, data = day3_tr2, family = binomial())
m_ovr_soft <- glm(mort_14day ~ PF_ratio_day3 + SOFA_24h + ovr_p_C2 + ovr_p_C3 + ovr_p_C4, data = day3_tr2, family = binomial())

# With PF+SOFA Day 3
m_mc_hard_d3  <- glm(mort_14day ~ PF_ratio_day3 + SOFA_day3 + hard_mc, data = day3_tr2, family = binomial())
m_mc_soft_d3  <- glm(mort_14day ~ PF_ratio_day3 + SOFA_day3 + mc_p_C2 + mc_p_C3 + mc_p_C4, data = day3_tr2, family = binomial())
m_ovr_hard_d3 <- glm(mort_14day ~ PF_ratio_day3 + SOFA_day3 + hard_ovr, data = day3_tr2, family = binomial())
m_ovr_soft_d3 <- glm(mort_14day ~ PF_ratio_day3 + SOFA_day3 + ovr_p_C2 + ovr_p_C3 + ovr_p_C4, data = day3_tr2, family = binomial())

# Class-only (for interest)
m_mc_hard_only  <- glm(mort_14day ~ hard_mc, data = day3_tr2, family = binomial())
m_mc_soft_only  <- glm(mort_14day ~ mc_p_C2 + mc_p_C3 + mc_p_C4, data = day3_tr2, family = binomial())
m_ovr_hard_only <- glm(mort_14day ~ hard_ovr, data = day3_tr2, family = binomial())
m_ovr_soft_only <- glm(mort_14day ~ ovr_p_C2 + ovr_p_C3 + ovr_p_C4, data = day3_tr2, family = binomial())

# =============================================================================
# 7) Predict on TEST + metrics (AUC/DeLong + calibration + Brier)
# =============================================================================
day3_te2 <- day3_te2 %>%
  mutate(
    p_pf            = predict(m_pf,           newdata = ., type="response"),
    p_base          = predict(m_base,         newdata = ., type="response"),
    p_base_d3       = predict(m_base_d3,      newdata = ., type="response"),
    
    p_mc_hard       = predict(m_mc_hard,      newdata = ., type="response"),
    p_mc_soft       = predict(m_mc_soft,      newdata = ., type="response"),
    p_ovr_hard      = predict(m_ovr_hard,     newdata = ., type="response"),
    p_ovr_soft      = predict(m_ovr_soft,     newdata = ., type="response"),
    
    p_mc_hard_d3    = predict(m_mc_hard_d3,   newdata = ., type="response"),
    p_mc_soft_d3    = predict(m_mc_soft_d3,   newdata = ., type="response"),
    p_ovr_hard_d3   = predict(m_ovr_hard_d3,  newdata = ., type="response"),
    p_ovr_soft_d3   = predict(m_ovr_soft_d3,  newdata = ., type="response"),
    
    p_mc_hard_only  = predict(m_mc_hard_only, newdata = ., type="response"),
    p_mc_soft_only  = predict(m_mc_soft_only, newdata = ., type="response"),
    p_ovr_hard_only = predict(m_ovr_hard_only,newdata = ., type="response"),
    p_ovr_soft_only = predict(m_ovr_soft_only,newdata = ., type="response")
  )

model_map <- list(
  "PF only"                 = "p_pf",
  "Baseline (PF+SOFA)"      = "p_base",
  "Baseline (PF+SOFA D3)"   = "p_base_d3",
  
  "Enhanced (MC hard)"      = "p_mc_hard",
  "Enhanced (MC probs)"     = "p_mc_soft",
  "Enhanced (OvR hard)"     = "p_ovr_hard",
  "Enhanced (OvR probs)"    = "p_ovr_soft",
  
  "Enhanced (MC hard D3)"   = "p_mc_hard_d3",
  "Enhanced (MC probs D3)"  = "p_mc_soft_d3",
  "Enhanced (OvR hard D3)"  = "p_ovr_hard_d3",
  "Enhanced (OvR probs D3)" = "p_ovr_soft_d3",
  
  "Class-only (MC hard)"    = "p_mc_hard_only",
  "Class-only (MC probs)"   = "p_mc_soft_only",
  "Class-only (OvR hard)"   = "p_ovr_hard_only",
  "Class-only (OvR probs)"  = "p_ovr_soft_only"
)

# AUC + ROC objects
auc_list <- lapply(model_map, function(col) auc_ci(day3_te2$y01, day3_te2[[col]]))
roc_base <- auc_list[["Baseline (PF+SOFA)"]]$roc
roc_base_d3 <- auc_list[["Baseline (PF+SOFA D3)"]]$roc
roc_pf   <- auc_list[["PF only"]]$roc

# Summary table
summary_tbl <- bind_rows(lapply(names(model_map), function(nm) {
  col <- model_map[[nm]]
  auc_obj <- auc_list[[nm]]
  cal <- calibration_stats(day3_te2$y01, day3_te2[[col]])
  
  # Determine which baseline to compare against
  # If model has "D3", compare to "Baseline (PF+SOFA D3)"
  # Else if model is "PF only" or "Baseline ...", handle specially
  # Else compare to "Baseline (PF+SOFA)"
  
  is_d3 <- grepl("D3", nm)
  
  p_val_base <- NA_real_
  if (nm == "PF only") {
    p_val_base <- NA_real_
  } else if (nm == "Baseline (PF+SOFA)") {
    p_val_base <- NA_real_
  } else if (nm == "Baseline (PF+SOFA D3)") {
    p_val_base <- NA_real_ # Or compare to Baseline (PF+SOFA)? Let's leave NA for now or compare to PF only?
                           # Actually, let's compare Baseline D3 to Baseline 24h? Or just leave NA.
                           # The user wants to compare using day 3 sofa.
  } else {
    # Enhanced or Class-only
    if (is_d3) {
      p_val_base <- delong_p(roc_base_d3, auc_obj$roc)
    } else {
      p_val_base <- delong_p(roc_base, auc_obj$roc)
    }
  }
  
  tibble(
    Model = nm,
    AUC = auc_obj$auc,
    AUC_lower_95CI = auc_obj$lo,
    AUC_upper_95CI = auc_obj$hi,
    AUC_formatted = fmt_auc(auc_obj),
    DeLong_p_vs_baseline = p_val_base,
    DeLong_p_vs_PF       = if (nm == "PF only") NA_real_ else delong_p(roc_pf, auc_obj$roc),
    Calibration_intercept = cal$cal_intercept,
    Calibration_slope     = cal$cal_slope,
    Brier_score    = cal$brier,
    AIC_train     = NA_real_,
    BIC_train     = NA_real_
  )
}))

# Fill AIC/BIC for models we fit
aic_map <- c(
  "PF only" = AIC(m_pf),
  "Baseline (PF+SOFA)" = AIC(m_base),
  "Baseline (PF+SOFA D3)" = AIC(m_base_d3),
  
  "Enhanced (MC hard)" = AIC(m_mc_hard),
  "Enhanced (MC probs)" = AIC(m_mc_soft),
  "Enhanced (OvR hard)" = AIC(m_ovr_hard),
  "Enhanced (OvR probs)" = AIC(m_ovr_soft),
  
  "Enhanced (MC hard D3)" = AIC(m_mc_hard_d3),
  "Enhanced (MC probs D3)" = AIC(m_mc_soft_d3),
  "Enhanced (OvR hard D3)" = AIC(m_ovr_hard_d3),
  "Enhanced (OvR probs D3)" = AIC(m_ovr_soft_d3),
  
  "Class-only (MC hard)" = AIC(m_mc_hard_only),
  "Class-only (MC probs)" = AIC(m_mc_soft_only),
  "Class-only (OvR hard)" = AIC(m_ovr_hard_only),
  "Class-only (OvR probs)" = AIC(m_ovr_soft_only)
)
bic_map <- c(
  "PF only" = BIC(m_pf),
  "Baseline (PF+SOFA)" = BIC(m_base),
  "Baseline (PF+SOFA D3)" = BIC(m_base_d3),
  
  "Enhanced (MC hard)" = BIC(m_mc_hard),
  "Enhanced (MC probs)" = BIC(m_mc_soft),
  "Enhanced (OvR hard)" = BIC(m_ovr_hard),
  "Enhanced (OvR probs)" = BIC(m_ovr_soft),
  
  "Enhanced (MC hard D3)" = BIC(m_mc_hard_d3),
  "Enhanced (MC probs D3)" = BIC(m_mc_soft_d3),
  "Enhanced (OvR hard D3)" = BIC(m_ovr_hard_d3),
  "Enhanced (OvR probs D3)" = BIC(m_ovr_soft_d3),
  
  "Class-only (MC hard)" = BIC(m_mc_hard_only),
  "Class-only (MC probs)" = BIC(m_mc_soft_only),
  "Class-only (OvR hard)" = BIC(m_ovr_hard_only),
  "Class-only (OvR probs)" = BIC(m_ovr_soft_only)
)

summary_tbl <- summary_tbl %>%
  mutate(
    AIC_train = aic_map[Model],
    BIC_train = bic_map[Model]
  )

write_csv(summary_tbl, file.path(root_dir, "Model_Comparison_Summary_TEST_withCal_Brier.csv"))

cat("\n=== FINAL SUMMARY (TEST) ===\n")
cat("Note: Class probabilities predicted using Day 0-3 physiology data\n")
cat("      Class predictor trained on TRAIN split only (80% of data)\n\n")
print(summary_tbl, n = Inf, width = Inf)

# =============================================================================
# 8) Decision curve net benefit (TEST)
# =============================================================================
# We'll include the main comparators + class-only as separate panels.
# (You can trim if you want fewer lines on the plot.)
pred_cols_main <- c("p_pf","p_base","p_mc_hard","p_mc_soft","p_ovr_hard","p_ovr_soft")
save_dca(day3_te2, outcome_col = "y01", pred_cols = pred_cols_main, out_prefix = "DCA_MAIN")

pred_cols_d3 <- c("p_pf","p_base_d3","p_mc_hard_d3","p_mc_soft_d3","p_ovr_hard_d3","p_ovr_soft_d3")
save_dca(day3_te2, outcome_col = "y01", pred_cols = pred_cols_d3, out_prefix = "DCA_MAIN_D3")

pred_cols_classonly <- c("p_mc_hard_only","p_mc_soft_only","p_ovr_hard_only","p_ovr_soft_only")
save_dca(day3_te2, outcome_col = "y01", pred_cols = pred_cols_classonly, out_prefix = "DCA_CLASS_ONLY")

# =============================================================================
# 9) ROC plot (TEST)
# =============================================================================
roc_df <- bind_rows(
  tibble(FPR = 1 - auc_list[["PF only"]]$roc$specificities,
         TPR = auc_list[["PF only"]]$roc$sensitivities, Model = "PF only"),
  tibble(FPR = 1 - auc_list[["Baseline (PF+SOFA)"]]$roc$specificities,
         TPR = auc_list[["Baseline (PF+SOFA)"]]$roc$sensitivities, Model = "Baseline"),
  tibble(FPR = 1 - auc_list[["Enhanced (MC hard)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (MC hard)"]]$roc$sensitivities, Model = "MC hard"),
  tibble(FPR = 1 - auc_list[["Enhanced (MC probs)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (MC probs)"]]$roc$sensitivities, Model = "MC probs"),
  tibble(FPR = 1 - auc_list[["Enhanced (OvR hard)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (OvR hard)"]]$roc$sensitivities, Model = "OvR hard"),
  tibble(FPR = 1 - auc_list[["Enhanced (OvR probs)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (OvR probs)"]]$roc$sensitivities, Model = "OvR probs"),
  
  tibble(FPR = 1 - auc_list[["Baseline (PF+SOFA D3)"]]$roc$specificities,
         TPR = auc_list[["Baseline (PF+SOFA D3)"]]$roc$sensitivities, Model = "Baseline D3"),
  tibble(FPR = 1 - auc_list[["Enhanced (MC hard D3)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (MC hard D3)"]]$roc$sensitivities, Model = "MC hard D3"),
  tibble(FPR = 1 - auc_list[["Enhanced (MC probs D3)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (MC probs D3)"]]$roc$sensitivities, Model = "MC probs D3"),
  tibble(FPR = 1 - auc_list[["Enhanced (OvR hard D3)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (OvR hard D3)"]]$roc$sensitivities, Model = "OvR hard D3"),
  tibble(FPR = 1 - auc_list[["Enhanced (OvR probs D3)"]]$roc$specificities,
         TPR = auc_list[["Enhanced (OvR probs D3)"]]$roc$sensitivities, Model = "OvR probs D3")
)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, colour = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "gray50") +
  coord_equal() +
  theme_minimal(base_size = 14) +
  labs(
    title = "ROC Curves (TEST): 14-day mortality",
    subtitle = sprintf("TEST n=%d; events=%d (%.1f%%)", nrow(day3_te2), sum(day3_te2$y01), 100*mean(day3_te2$y01)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme(legend.position = "bottom", legend.title = element_blank())

ggsave(file.path(root_dir, "ROC_TEST.png"), p_roc, width = 10, height = 7, dpi = 300)

cat("\n✅ Outputs saved to: ", root_dir, "\n", sep="")
cat(" - Model_Comparison_Summary_TEST_withCal_Brier.csv\n")
cat(" - DCA_MAIN_DCA.png + DCA_MAIN_DCA_net_benefit.csv\n")
cat(" - DCA_MAIN_D3_DCA.png + DCA_MAIN_D3_DCA_net_benefit.csv\n")
cat(" - DCA_CLASS_ONLY_DCA.png + DCA_CLASS_ONLY_DCA_net_benefit.csv\n")
cat(" - ROC_TEST.png\n")
cat("\n=== Done ===\n")


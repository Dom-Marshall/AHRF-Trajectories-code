# 0. Packages -----------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(xgboost)
  library(pROC)        # AUC
  library(caret)       # train/test split & metrics
  library(ggplot2)
  library(zoo)         # LOCF
  library(ggridges)    # importance visuals
})

# ---- Consistent class colours (C1..C4) --------------------------------------
class_levels     <- paste0("C", 1:4)
class_cols_named <- c(
  "C1" = "deeppink",
  "C2" = "deepskyblue",
  "C3" = "forestgreen",
  "C4" = "orangered"
)

# 1. Load & preprocess static data -------------------------------------------
prob_mim  <- read.csv("Data/pprob_MIMIC.csv")     %>% select(stay_id, class)
other_raw <- read.csv("Data/mimic_static_var.csv") %>%
  mutate(sex = ifelse(gender == 'M', 1, 0)) %>%
  select(stay_id, admission_age, sex)
vars28    <- read.csv("Data/mimic_dynamic_var.csv")

# 2. Utility: train/test split ------------------------------------------------
default_split <- function(df) {
  createDataPartition(df$class, p = 0.8, list = FALSE)
}

# 3. Feature builders ---------------------------------------------------------
measure_vars <- c(
  "norad_vasorate", "avg_lactate", "locf_bicarbonate",
  "avg_peak_insp_pressure", "avg_peep", "avg_pao2fio2ratio",
  "locf_creatinine", "avg_pco2", "avg_minute_volume",
  "avg_resp_rate"
)

build_feats_static <- function(d) {
  dyn <- vars28 %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = days_from_start,
      values_from = all_of(measure_vars),
      names_glue = "day{days_from_start}_{.value}"
    ) %>%
    mutate(across(contains("pao2fio2ratio"), ~ ifelse(. > 470, NA, .)))
  
  prob_mim %>%
    inner_join(other_raw, by = "stay_id") %>%
    left_join(dyn, by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}

build_feats_traj <- function(d) {
  long <- vars28 %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_longer(cols = all_of(measure_vars), names_to = "var", values_to = "val")
  
  sum_dt <- long %>%
    group_by(stay_id, var) %>%
    arrange(days_from_start) %>%
    summarise(
      first    = first(val),
      last     = {v <- zoo::na.locf(val, na.rm = FALSE); v[length(v)]},
      slope    = if (d > 0) (last - first) / d else NA_real_,
      variance = var(val, na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = var,
      values_from = c(first, last, slope, variance),
      names_glue = "{.value}_{var}"
    )
  
  prob_mim %>%
    inner_join(other_raw, by = "stay_id") %>%
    left_join(sum_dt, by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}

build_feats_last <- function(d) {
  long <- vars28 %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_longer(cols = all_of(measure_vars), names_to = "var", values_to = "val")
  
  last_dt <- long %>%
    group_by(stay_id, var) %>%
    arrange(days_from_start) %>%
    summarise(
      last = {v <- zoo::na.locf(val, na.rm = FALSE); v[length(v)]},
      .groups = "drop"
    ) %>%
    pivot_wider(
      id_cols    = stay_id,
      names_from = var,
      values_from = last,
      names_glue = "last_{var}"
    )
  
  prob_mim %>%
    inner_join(other_raw, by = "stay_id") %>%
    left_join(last_dt, by = "stay_id") %>%
    mutate(class = factor(paste0("C", class), levels = class_levels))
}

# 4. Split indices -----------------------------------------------------------
set.seed(1234)
all_static7 <- build_feats_static(7)
split_idx   <- default_split(all_static7)

# 5. Output directory & plotting helpers -------------------------------------
out_dir <- "mimic_class_prediction_vs_timev3"
if (!dir.exists(out_dir)) dir.create(out_dir)

if (!exists("theme_custom")) theme_custom <- theme_bw()
if (!exists("save_plot")) {
  save_plot <- function(plot, filename, width = 7, height = 5) {
    ggsave(file.path(out_dir, filename), plot = plot, width = width, height = height)
  }
}

# 6. XGBoost parameter sets ---------------------------- (determined after gridsearch) ----------------
static_xgb_multi_params <- list(
  booster          = "gbtree",
  objective        = "multi:softprob",
  num_class        = length(levels(all_static7$class)),
  eval_metric      = "mlogloss",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 1.0
)
static_xgb_bin_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 1.0
)

traj_xgb_multi_params <- list(
  booster          = "gbtree",
  objective        = "multi:softprob",
  num_class        = length(levels(all_static7$class)),
  eval_metric      = "mlogloss",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.7,
  colsample_bytree = 1.0
)
traj_xgb_bin_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.7,
  colsample_bytree = 1.0
)

last_xgb_multi_params <- list(
  booster          = "gbtree",
  objective        = "multi:softprob",
  num_class        = length(levels(all_static7$class)),
  eval_metric      = "mlogloss",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 1.0
)
last_xgb_bin_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 1.0
)

metrics <- c("AUC","Accuracy","Sensitivity","Specificity")

# ----------------------------------------------------------------------------
# Calibration helpers --------------------------------------------------------
ece_brier <- function(prob, truth, k = 10) {
  cuts   <- seq(0, 1, length.out = k + 1)
  binfac <- factor(cut(prob, cuts, include.lowest = TRUE, labels = FALSE),
                   levels = 1:k)
  exp    <- tapply(prob , binfac, mean,   na.rm = TRUE)
  obs    <- tapply(truth, binfac, mean,   na.rm = TRUE)
  counts <- tapply(truth, binfac, length)
  exp[is.na(exp)]       <- 0
  obs[is.na(obs)]       <- 0
  counts[is.na(counts)] <- 0
  ece_val   <- if (sum(counts) > 0)
    sum(abs(obs - exp) * counts) / sum(counts) else NA_real_
  brier_val <- mean((prob - truth)^2)
  df <- tibble(
    bin       = seq_len(k),
    exp       = exp,
    obs       = obs,
    bin_count = counts
  )
  list(ece = ece_val, brier = brier_val, df = df)
}

save_reliability <- function(cal_df, title, file_stub) {
  cal_df %>%
    filter(bin_count > 0) %>%
    ggplot(aes(exp, obs, size = bin_count)) +
    geom_point(alpha = .7) +
    geom_abline(linetype = "dashed") +
    scale_size_continuous(range = c(1, 6)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = title,
      x     = "Predicted",
      y     = "Observed",
      size  = "n"
    ) +
    theme_bw() -> p
  
  ggsave(
    filename = file.path(out_dir, file_stub),
    plot     = p,
    width    = 5,
    height   = 5
  )
}

# xgboost 3.x compatibility: best_iteration moved to cv$early_stop$best_iteration
get_best_nrounds <- function(cv, fallback = 300L) {
  bi <- cv$early_stop$best_iteration
  if (!is.null(bi) && length(bi) > 0L) as.integer(bi) else fallback
}

# ----------------------------------------------------------------------------
# Helper to train & return importance & preds (modularized) ------------------
fit_xgb <- function(train_mat, train_lab, test_mat, params,
                    multiclass = TRUE, num_class = NULL) {
  dtrain <- xgb.DMatrix(train_mat, label = train_lab)
  cv  <- xgb.cv(params, dtrain, nrounds = 300, nfold = 5,
                early_stopping_rounds = 20, verbose = 0)
  model <- xgb.train(params, dtrain, nrounds = get_best_nrounds(cv), verbose = 0)
  preds <- predict(model, xgb.DMatrix(test_mat))
  if (multiclass) {
    preds <- matrix(preds, ncol = num_class, byrow = TRUE)
  }
  list(model = model, preds = preds)
}

# ----------------------------------------------------------------------------
# Section A – Static cumulative features -------------------------------------
static_days           <- 0:7
results_static_multi  <- tibble()
results_static_bin    <- tibble()
importance_static_mc  <- list()
importance_static_bin <- list()

for (d in static_days) {
  feats <- build_feats_static(d)
  train <- feats[split_idx, ]
  test  <- feats[-split_idx, ]
  preds <- setdiff(names(train), c("stay_id","class"))
  
  ## Multiclass
  multi_out <- fit_xgb(
    as.matrix(train[, preds]),
    as.numeric(train$class) - 1,
    as.matrix(test[,  preds]),
    static_xgb_multi_params,
    multiclass = TRUE,
    num_class  = length(class_levels)
  )
  mc    <- multi_out$model
  pmat  <- multi_out$preds
  importance_static_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  
  plab  <- factor(class_levels[max.col(pmat)], levels = class_levels)
  cm_mc <- confusionMatrix(plab, test$class)
  
  # Multiclass calibration
  Y_onehot <- model.matrix(~ class - 1, data = test)
  col_order <- paste0("class", class_levels)
  if (!all(colnames(Y_onehot) == col_order)) {
    Y_onehot <- Y_onehot[, col_order, drop = FALSE]
  }
  brier_mc <- mean((pmat - Y_onehot)^2)
  cal_mc   <- ece_brier(as.numeric(pmat), as.numeric(Y_onehot))
  save_reliability(
    cal_mc$df,
    sprintf("Static multiclass Day %d", d),
    sprintf("calib_static_multiclass_day%02d.png", d)
  )
  results_static_multi <- bind_rows(
    results_static_multi,
    tibble(
      day      = d,
      Accuracy = cm_mc$overall["Accuracy"],
      Brier    = brier_mc,
      ECE      = cal_mc$ece
    )
  )
  
  ## Binary one-vs-rest
  for (cls in class_levels) {
    ytr  <- as.numeric(train$class == cls)
    yte  <- as.numeric(test$class  == cls)
    bin_out <- fit_xgb(
      as.matrix(train[, preds]), ytr,
      as.matrix(test[,  preds]), static_xgb_bin_params,
      multiclass = FALSE
    )
    bin   <- bin_out$model
    prob  <- bin_out$preds
    importance_static_bin[[as.character(d)]][[cls]] <-
      xgb.importance(preds, model = bin)
    
    truth <- factor(yte, levels = c(0,1))
    pred  <- factor(as.numeric(prob > .5), levels = c(0,1))
    roc_o <- roc(as.numeric(truth), prob, quiet = TRUE)
    cm_b  <- confusionMatrix(pred, truth)
    
    cal <- ece_brier(prob, yte)
    save_reliability(
      cal$df,
      sprintf("Static %s Day %d", cls, d),
      sprintf("calib_static_%s_day%02d.png", cls, d)
    )
    
    results_static_bin <- bind_rows(
      results_static_bin,
      tibble(
        day         = d,
        class       = cls,
        AUC         = as.numeric(auc(roc_o)),
        Accuracy    = cm_b$overall["Accuracy"],
        Sensitivity = cm_b$byClass["Sensitivity"],
        Specificity = cm_b$byClass["Specificity"],
        Brier       = cal$brier,
        ECE         = cal$ece
      )
    )
  }
}

# ----------------------------------------------------------------------------
# Plots / CSVs for Section A --------------------------------------------------
save_plot(
  ggplot(results_static_multi, aes(day, Accuracy)) +
    geom_line() + geom_point() +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = "Static Multiclass Accuracy", x = "Day", y = "Accuracy") +
    theme_custom,
  "static_multiclass_accuracy.png"
)
for (m in metrics) {
  save_plot(
    ggplot(results_static_bin,
           aes(day, .data[[m]], colour = class, group = class)) +
      geom_line() + geom_point() +
      scale_y_continuous(limits = c(0,1)) +
      scale_colour_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste0("Static ", m, " (OvR)"),
           x = "Day", y = m) +
      theme_custom,
    paste0("static_binary_", tolower(m), ".png")
  )
}
results_static_multi %>% write.csv(file.path(out_dir,"static_multiclass_metrics.csv"), row.names = FALSE)
results_static_bin   %>% write.csv(file.path(out_dir,"static_binary_metrics.csv")  , row.names = FALSE)

# ----------------------------------------------------------------------------
# Section B – Trajectory summary features ------------------------------------
traj_days            <- 1:7
results_traj_multi   <- tibble()
results_traj_bin     <- tibble()
importance_traj_mc   <- list()
importance_traj_bin  <- list()

for (d in traj_days) {
  feats <- build_feats_traj(d)
  train <- feats[split_idx, ];  test <- feats[-split_idx, ]
  preds <- setdiff(names(train), c("stay_id","class"))
  
  # Multiclass ---------------------------------------------------------------
  dmat <- xgb.DMatrix(as.matrix(train[, preds]), label = as.numeric(train$class) - 1)
  cv   <- xgb.cv(traj_xgb_multi_params, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
  mc   <- xgb.train(traj_xgb_multi_params, dmat, nrounds = get_best_nrounds(cv), verbose = 0)
  importance_traj_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  
  pmat <- predict(mc, xgb.DMatrix(as.matrix(test[, preds]))) %>%
    matrix(ncol = length(levels(train$class)), byrow = TRUE)
  plab <- factor(levels(train$class)[max.col(pmat)], levels = levels(train$class))
  cm_mc<- confusionMatrix(plab, test$class)
  
  # Multiclass calibration
  Y_onehot <- model.matrix(~ class - 1, data = test)
  col_order <- paste0("class", class_levels)
  if (!all(colnames(Y_onehot) == col_order)) {
    Y_onehot <- Y_onehot[, col_order, drop = FALSE]
  }
  brier_mc <- mean((pmat - Y_onehot)^2)
  cal_mc   <- ece_brier(as.numeric(pmat), as.numeric(Y_onehot))
  save_reliability(
    cal_mc$df,
    sprintf("Trajectory multiclass 0-%d", d),
    sprintf("cal_traj_multiclass_0-%02d.png", d)
  )
  results_traj_multi <- bind_rows(
    results_traj_multi,
    tibble(window = paste0("0-", d),
           Accuracy = cm_mc$overall["Accuracy"],
           Brier    = brier_mc,
           ECE      = cal_mc$ece)
  )
  
  # Binary -------------------------------------------------------------------
  importance_traj_bin[[as.character(d)]] <- list()
  
  for (cls in levels(train$class)) {
    ytr  <- as.numeric(train$class == cls)
    yte  <- as.numeric(test$class  == cls)
    bmat <- xgb.DMatrix(as.matrix(train[, preds]), label = ytr)
    cvb  <- xgb.cv(traj_xgb_bin_params, bmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
    bin  <- xgb.train(traj_xgb_bin_params, bmat, nrounds = get_best_nrounds(cvb), verbose = 0)
    importance_traj_bin[[as.character(d)]][[cls]] <- xgb.importance(preds, model = bin)
    
    prob  <- predict(bin, xgb.DMatrix(as.matrix(test[, preds])))
    truth <- factor(yte, levels = c(0,1))
    pred  <- factor(as.numeric(prob > .5), levels = c(0,1))
    roc_o <- roc(as.numeric(truth), prob, quiet = TRUE)
    cm_b  <- confusionMatrix(pred, truth)
    
    cal <- ece_brier(prob, yte)
    save_reliability(
      cal$df,
      sprintf("Trajectory  %s  0-%d", cls, d),
      sprintf("cal_traj_%s_0-%02d.png", cls, d)
    )
    
    results_traj_bin <- bind_rows(
      results_traj_bin,
      tibble(
        window      = paste0("0-", d),
        class       = factor(cls, levels = class_levels),
        AUC         = as.numeric(auc(roc_o)),
        Accuracy    = cm_b$overall["Accuracy"],
        Sensitivity = cm_b$byClass["Sensitivity"],
        Specificity = cm_b$byClass["Specificity"],
        Brier       = cal$brier,
        ECE         = cal$ece
      )
    )
  }
}

# ----------------------------------------------------------------------------
# Section C – Last-value features --------------------------------------------
last_days            <- 0:7
results_last_multi   <- tibble()
results_last_bin     <- tibble()
importance_last_mc   <- list()
importance_last_bin  <- list()

for (d in last_days) {
  feats <- build_feats_last(d)
  train <- feats[split_idx, ];  test <- feats[-split_idx, ]
  preds <- setdiff(names(train), c("stay_id","class"))
  
  # Multiclass ---------------------------------------------------------------
  dmat <- xgb.DMatrix(as.matrix(train[, preds]), label = as.numeric(train$class) - 1)
  cv   <- xgb.cv(last_xgb_multi_params, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
  mc   <- xgb.train(last_xgb_multi_params, dmat, nrounds = get_best_nrounds(cv), verbose = 0)
  importance_last_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  
  pmat <- predict(mc, xgb.DMatrix(as.matrix(test[, preds]))) %>%
    matrix(ncol = length(levels(train$class)), byrow = TRUE)
  plab <- factor(levels(train$class)[max.col(pmat)], levels = levels(train$class))
  cm_mc<- confusionMatrix(plab, test$class)
  
  # Multiclass calibration
  Y_onehot <- model.matrix(~ class - 1, data = test)
  col_order <- paste0("class", class_levels)
  if (!all(colnames(Y_onehot) == col_order)) {
    Y_onehot <- Y_onehot[, col_order, drop = FALSE]
  }
  brier_mc <- mean((pmat - Y_onehot)^2)
  cal_mc   <- ece_brier(as.numeric(pmat), as.numeric(Y_onehot))
  save_reliability(
    cal_mc$df,
    sprintf("Last-value multiclass Day %d", d),
    sprintf("cal_last_multiclass_day%02d.png", d)
  )
  
  results_last_multi <- bind_rows(
    results_last_multi,
    tibble(day = d,
           Accuracy = cm_mc$overall["Accuracy"],
           Brier    = brier_mc,
           ECE      = cal_mc$ece)
  )
  
  # Binary -------------------------------------------------------------------
  importance_last_bin[[as.character(d)]] <- list()
  
  for (cls in levels(train$class)) {
    ytr  <- as.numeric(train$class == cls)
    yte  <- as.numeric(test$class  == cls)
    bmat <- xgb.DMatrix(as.matrix(train[, preds]), label = ytr)
    cvb  <- xgb.cv(last_xgb_bin_params, bmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
    bin  <- xgb.train(last_xgb_bin_params, bmat, nrounds = get_best_nrounds(cvb), verbose = 0)
    importance_last_bin[[as.character(d)]][[cls]] <- xgb.importance(preds, model = bin)
    
    prob  <- predict(bin, xgb.DMatrix(as.matrix(test[, preds])))
    truth <- factor(yte, levels = c(0,1))
    pred  <- factor(as.numeric(prob > .5), levels = c(0,1))
    roc_o <- roc(as.numeric(truth), prob, quiet = TRUE)
    cm_b  <- confusionMatrix(pred, truth)
    
    cal <- ece_brier(prob, yte)
    save_reliability(
      cal$df,
      sprintf("Last-value  %s  Day %d", cls, d),
      sprintf("cal_last_%s_day%02d.png", cls, d)
    )
    
    results_last_bin <- bind_rows(
      results_last_bin,
      tibble(
        day         = d,
        class       = factor(cls, levels = class_levels),
        AUC         = as.numeric(auc(roc_o)),
        Accuracy    = cm_b$overall["Accuracy"],
        Sensitivity = cm_b$byClass["Sensitivity"],
        Specificity = cm_b$byClass["Specificity"],
        Brier       = cal$brier,
        ECE         = cal$ece
      )
    )
  }
}

# ----------------------------------------------------------------------------
# --  Outputs & summary plots (multiclass plots kept as-is) -------------------
save_plot(
  ggplot(results_traj_multi, aes(window, Accuracy, group = 1)) +
    geom_line() + geom_point() +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = "Trajectory Multiclass Accuracy", x = "Window", y = "Accuracy") +
    theme_custom,
  "traj_multiclass_accuracy.png"
)
save_plot(
  ggplot(results_last_multi, aes(day, Accuracy, group = 1)) +
    geom_line() + geom_point() +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = "Last-value Multiclass Accuracy", x = "Day", y = "Accuracy") +
    theme_custom,
  "last_multiclass_accuracy.png"
)

for (m in metrics) {
  # trajectory (OvR)
  save_plot(
    ggplot(results_traj_bin,
           aes(window, .data[[m]], colour = class, group = class)) +
      geom_line() + geom_point() +
      scale_y_continuous(limits = c(0,1)) +
      scale_colour_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste0("Trajectory ", m, " (OvR)"),
           x = "Window", y = m) +
      theme_custom,
    paste0("traj_binary_", tolower(m), ".png")
  )
  # last-value (OvR)
  save_plot(
    ggplot(results_last_bin,
           aes(day, .data[[m]], colour = class, group = class)) +
      geom_line() + geom_point() +
      scale_y_continuous(limits = c(0,1)) +
      scale_colour_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste0("Last-value ", m, " (OvR)"),
           x = "Day", y = m) +
      theme_custom,
    paste0("last_binary_", tolower(m), ".png")
  )
}

# write CSVs with the new calibration columns included
results_traj_multi %>% write.csv(file.path(out_dir,"traj_multiclass_metrics.csv"), row.names = FALSE)
results_traj_bin   %>% write.csv(file.path(out_dir,"traj_binary_metrics.csv")  , row.names = FALSE)
results_last_multi %>% write.csv(file.path(out_dir,"last_multiclass_metrics.csv"), row.names = FALSE)
results_last_bin   %>% write.csv(file.path(out_dir,"last_binary_metrics.csv")  , row.names = FALSE)

message("Finished – all outputs (including calibration) are in: ", normalizePath(out_dir))

# ---------------------------------------------------------------------------
# Variable–profile plots (days 0-14) with n≥10 filter ------------------------
prof_out <- file.path(out_dir, "var_profiles2")
if (!dir.exists(prof_out)) dir.create(prof_out)

var_long <- vars28 %>%
  filter(days_from_start %in% 0:14) %>%                         # first 14 ICU days
  select(stay_id, days_from_start, all_of(measure_vars)) %>%
  pivot_longer(-c(stay_id, days_from_start),
               names_to = "variable", values_to = "value") %>%
  left_join(prob_mim %>% mutate(class = factor(paste0("C", class), levels = class_levels)),
            by = "stay_id")

# Mean ± SE profiles
for (v in measure_vars) {
  tab <- var_long %>%
    filter(variable == v) %>%
    group_by(class, days_from_start) %>%
    summarise(
      n_total   = n(),
      n_miss    = sum(is.na(value)),
      n_keep    = n_total - n_miss,
      p_miss    = n_miss / n_total,
      mean_val  = mean(value, na.rm = TRUE),
      se_val    = sd(value,  na.rm = TRUE) / sqrt(n_keep),
      .groups   = "drop"
    ) %>%
    mutate(
      mean_val = if_else(n_keep < 10, NA_real_, mean_val),
      se_val   = if_else(n_keep < 10, NA_real_, se_val),
      p_miss   = if_else(n_keep < 10, NA_real_, p_miss)
    ) %>%
    filter(!is.na(mean_val))
  if (nrow(tab) == 0) next
  
  p <- ggplot(tab, aes(days_from_start, mean_val,
                       colour = class, group = class)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = mean_val - se_val,
                    ymax = mean_val + se_val,
                    fill  = class),
                alpha = 0.15, colour = NA) +
    geom_point(aes(size = p_miss), alpha = 0.7) +
    ## NEW — fixed x-axis 0–14, ticks every 2
    scale_x_continuous(limits = c(0, 14), breaks = seq(0, 14, 2)) +
    scale_color_manual(values = class_cols_named, drop = FALSE) +
    scale_fill_manual(values = class_cols_named, guide = "none", drop = FALSE) +
    scale_size_continuous(
      range  = c(0.5, 4),
      breaks = c(0, 0.25, 0.5, 0.75),
      labels = scales::percent_format(accuracy = 1)
    ) +
    labs(title = v,
         x = "ICU Day",
         y = "Mean (± SE)",
         size = "% missing") +
    ## NEW — legend now at bottom
    theme_custom + theme(legend.position = "bottom")
  
  ggsave(file.path(prof_out, paste0(v, " 14day.png")),
         p, width = 9, height = 6)
}
# Median and IQR profiles
for (v in measure_vars) {
  tab <- var_long %>%
    filter(variable == v) %>%
    group_by(class, days_from_start) %>%
    summarise(
      n_total  = n(),
      n_miss   = sum(is.na(value)),
      n_keep   = n_total - n_miss,
      p_miss   = n_miss / n_total,
      med_val  = median(value, na.rm = TRUE),
      q25      = quantile(value, 0.25, na.rm = TRUE),
      q75      = quantile(value, 0.75, na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    mutate(
      med_val = if_else(n_keep < 10, NA_real_, med_val),
      q25     = if_else(n_keep < 10, NA_real_, q25),
      q75     = if_else(n_keep < 10, NA_real_, q75),
      p_miss  = if_else(n_keep < 10, NA_real_, p_miss)
    ) %>%
    filter(!is.na(med_val))
  if (nrow(tab) == 0) next
  
  p <- ggplot(tab, aes(days_from_start, med_val,
                       colour = class, group = class)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = q25, ymax = q75, fill = class),
                alpha = 0.15, colour = NA) +
    geom_point(aes(size = p_miss), alpha = 0.7) +
    scale_color_manual(values = class_cols_named, drop = FALSE) +
    scale_fill_manual(values = class_cols_named, guide = "none", drop = FALSE) +
    scale_size_continuous(
      range  = c(0.5, 4),
      breaks = c(0, 0.25, 0.5, 0.75),
      labels = scales::percent_format(accuracy = 1)
    ) +
    labs(title = v,
         x = "ICU Day",
         y = "Median (IQR)",
         size = "% missing") +
    theme_custom
  
  ggsave(file.path(prof_out, paste0(v, " median 14day.png")),
         plot   = p, width  = 9, height = 6)
}

# ------------------------------- Done ----------------------------------------
message("Finished. Outputs written to: ", normalizePath(out_dir, winslash = "/"))

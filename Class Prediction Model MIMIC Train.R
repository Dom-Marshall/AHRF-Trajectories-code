# 0. Packages -----------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(xgboost)
  library(pROC)        # for AUC
  library(caret)       # for data split and metrics
  library(ggplot2)
  library(zoo)         # for LOCF
  library(ggridges)    # ridgeline for importance visuals
})

# ---- Consistent class colours (C1..C4) --------------------------------------
class_levels     <- paste0("C", 1:4)
class_cols_named <- c("C1" = "deeppink",
                      "C2" = "deepskyblue",
                      "C3" = "forestgreen",
                      "C4" = "orangered")

# 1. Load & preprocess static data -------------------------------------------
prob_mim  <- read.csv("Data/pprob_MIMIC.csv") %>% select(stay_id, class) # A table with stay_id and assigned class
other_raw <- read.csv("Data/MIMIC_baseline_char.csv") %>%
  mutate(sex = ifelse(gender == 'M', 1, 0)) %>%
  select(stay_id, admission_age, sex)
vars28    <- read.csv("Data/MIMIC_28d_dyanamic_var.csv")

# 2. Utility: train/test split ------------------------------------------------
default_split <- function(df) {
  createDataPartition(df$class, p = 0.8, list = FALSE)
}

# 3. Feature builders --------------------------------------------------------
measure_vars <- c(
  "norad_vasorate", "avg_lactate", "locf_bicarbonate",
  "avg_peak_insp_pressure", "avg_peep", "avg_pao2fio2ratio",
  "locf_creatinine", "avg_pco2", "avg_minute_volume",
  "avg_resp_rate"
)

# 3A. Static cumulative features up to day d
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

# 3B. Trajectory summary (first, last, slope, variance)
build_feats_traj <- function(d) {
  long <- vars28 %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_longer(
      cols = all_of(measure_vars),
      names_to = "var",
      values_to = "val"
    )
  sum_dt <- long %>%
    group_by(stay_id, var) %>%
    arrange(days_from_start) %>%
    summarise(
      first    = first(val),
      last     = {v <- zoo::na.locf(val, na.rm = FALSE); v[length(v)]},
      slope    = if (d > 0) (last - first) / d else NA_real_,
      variance = var(val, na.rm = TRUE),
      .groups = 'drop'
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

# 3C. Last available value on day d (LOCF)
build_feats_last <- function(d) {
  long <- vars28 %>%
    filter(days_from_start %in% 0:d) %>%
    pivot_longer(
      cols = all_of(measure_vars),
      names_to = "var",
      values_to = "val"
    )
  last_dt <- long %>%
    group_by(stay_id, var) %>%
    arrange(days_from_start) %>%
    summarise(
      last = {v <- zoo::na.locf(val, na.rm = FALSE); v[length(v)]},
      .groups = 'drop'
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
all_static7 <- build_feats_static(7)
split_idx   <- default_split(all_static7)

# 5. Output directory --------------------------------------------------------
out_dir <- "mimic_class_prediction_vs_time"
if (!dir.exists(out_dir)) dir.create(out_dir)

# 6. XGBoost parameter sets --------------------------------------------------
xgb_multi_params <- list(
  booster          = "gbtree",
  objective        = "multi:softprob",
  num_class        = length(levels(all_static7$class)),
  eval_metric      = "mlogloss",
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)
xgb_bin_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)
metrics <- c("AUC","Accuracy","Sensitivity","Specificity")

# Custom theme ---------------------------------------------------------------
theme_custom <- theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'),
    axis.title = element_text(size = 12)
  )

# Helper to save plots ------------------------------------------------------
save_plot <- function(plot_obj, filename) {
  ggsave(
    filename = file.path(out_dir, filename),
    plot     = plot_obj,
    width    = 8, height = 6
  )
}

# ----------------------------------------------------------------------------
# Section A: Static Cumulative Features --------------------------------------
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
  
  # Multiclass model
  dmat  <- xgb.DMatrix(data = as.matrix(train[, preds]), label = as.numeric(train$class) - 1)
  cv    <- xgb.cv(xgb_multi_params, dmat, nrounds = 300, nfold = 5,
                  early_stopping_rounds = 20, verbose = FALSE)
  mc    <- xgb.train(xgb_multi_params, dmat, nrounds = cv$best_iteration, verbose = FALSE)
  importance_static_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  # Evaluate
  pmat  <- predict(mc, xgb.DMatrix(as.matrix(test[, preds]))) %>%
    matrix(ncol = length(levels(train$class)), byrow = TRUE)
  plab  <- factor(levels(train$class)[max.col(pmat)], levels = levels(train$class))
  cm_mc <- confusionMatrix(plab, test$class)
  results_static_multi <- bind_rows(results_static_multi,
                                    tibble(day = d, Accuracy = cm_mc$overall['Accuracy']))
  
  # Binary one-vs-rest models
  importance_static_bin[[as.character(d)]] <- list()
  for (cls in levels(train$class)) {
    ytr    <- as.numeric(train$class == cls)
    bmat   <- xgb.DMatrix(data = as.matrix(train[, preds]), label = ytr)
    cvb    <- xgb.cv(xgb_bin_params, bmat, nrounds = 300, nfold = 5,
                     early_stopping_rounds = 20, verbose = FALSE)
    bin    <- xgb.train(xgb_bin_params, bmat, nrounds = cvb$best_iteration, verbose = FALSE)
    importance_static_bin[[as.character(d)]][[cls]] <- xgb.importance(preds, model = bin)
    prob   <- predict(bin, xgb.DMatrix(as.matrix(test[, preds])))
    pred   <- factor(as.numeric(prob > 0.5), levels = c(0,1))
    truth  <- factor(as.numeric(test$class == cls), levels = c(0,1))
    roc_o  <- roc(as.numeric(truth), prob, quiet = TRUE)
    cm_b   <- confusionMatrix(pred, truth)
    
    results_static_bin <- bind_rows(
      results_static_bin,
      tibble(
        day         = d,
        class       = factor(cls, levels = class_levels),
        AUC         = as.numeric(auc(roc_o)),
        Accuracy    = cm_b$overall['Accuracy'],
        Sensitivity = cm_b$byClass['Sensitivity'],
        Specificity = cm_b$byClass['Specificity']
      )
    )
  }
}

# Save static plots
save_plot(
  ggplot(results_static_multi, aes(day, Accuracy)) +
    geom_line(group = 1) + geom_point() +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = 'Static Multiclass Accuracy', x = 'Day', y = 'Accuracy') + theme_custom,
  'static_multiclass_accuracy.png'
)
for (m in metrics) {
  save_plot(
    ggplot(results_static_bin, aes(x = day, y = .data[[m]], color = class, group = class)) +
      geom_line() + geom_point() +
      scale_y_continuous(limits = c(0,1)) +
      scale_color_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste('Static', m, 'One-vs-Rest'), x = 'Day', y = m) + theme_custom,
    paste0('static_binary_', tolower(m), '.png')
  )
}
# CSVs & importance heat-maps for Section A
results_static_multi %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'static_multiclass_metrics.csv'), row.names = FALSE)
results_static_bin %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'static_binary_metrics.csv'), row.names = FALSE)

# multiclass importance heat-map
topA <- map_dfr(static_days, ~importance_static_mc[[as.character(.x)]] %>% mutate(day=.x))
save_plot(
  ggplot(topA %>% group_by(day) %>% slice_max(Gain, n = 10),
         aes(day, reorder(Feature, Gain), fill = Gain)) +
    geom_tile() +
    labs(title = 'Static Multiclass Feature Importance', x = 'Day', y = 'Feature') +
    theme_custom,
  'static_mc_feature_importance.png'
)
# binary heat-maps
for (cls in levels(results_static_bin$class)) {
  dfA <- map_dfr(static_days,
                 ~importance_static_bin[[as.character(.x)]][[cls]] %>% mutate(day=.x))
  save_plot(
    ggplot(dfA %>% group_by(day) %>% slice_max(Gain, n = 10),
           aes(day, reorder(Feature, Gain), fill = Gain)) +
      geom_tile() +
      labs(title = paste('Static Binary Feature Importance -', cls),
           x = 'Day', y = 'Feature') + theme_custom,
    paste0('static_imp_binary_', cls, '.png')
  )
}

# ----------------------------------------------------------------------------
# Section B: Trajectory Summary Features -------------------------------------
traj_days <- 1:7
results_traj_multi <- tibble()
results_traj_bin   <- tibble()
importance_traj_mc <- list()
importance_traj_bin<- list()

for (d in traj_days) {
  feats <- build_feats_traj(d)
  train <- feats[split_idx, ]; test <- feats[-split_idx, ]
  preds <- setdiff(names(train), c('stay_id','class'))
  
  # Multiclass
  dmat  <- xgb.DMatrix(as.matrix(train[, preds]), label = as.numeric(train$class) - 1)
  cv    <- xgb.cv(xgb_multi_params, dmat, nrounds = 300, nfold = 5,
                  early_stopping_rounds = 20, verbose = FALSE)
  mc    <- xgb.train(xgb_multi_params, dmat, nrounds = cv$best_iteration, verbose = FALSE)
  importance_traj_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  pmat  <- predict(mc, xgb.DMatrix(as.matrix(test[, preds]))) %>%
    matrix(ncol = length(levels(train$class)), byrow = TRUE)
  plab  <- factor(levels(train$class)[max.col(pmat)], levels = levels(train$class))
  cm_mc <- confusionMatrix(plab, test$class)
  results_traj_multi <- bind_rows(
    results_traj_multi,
    tibble(window = paste0('0-', d), Accuracy = cm_mc$overall['Accuracy'])
  )
  
  # Binary
  importance_traj_bin[[as.character(d)]] <- list()
  for (cls in levels(train$class)) {
    ytr  <- as.numeric(train$class == cls)
    bmat <- xgb.DMatrix(as.matrix(train[, preds]), label = ytr)
    cvb  <- xgb.cv(xgb_bin_params, bmat, nrounds = 300, nfold = 5,
                   early_stopping_rounds = 20, verbose = FALSE)
    bin  <- xgb.train(xgb_bin_params, bmat, nrounds = cvb$best_iteration, verbose = FALSE)
    importance_traj_bin[[as.character(d)]][[cls]] <- xgb.importance(preds, model = bin)
    prob <- predict(bin, xgb.DMatrix(as.matrix(test[, preds])))
    pred <- factor(as.numeric(prob>0.5), levels = c(0,1))
    truth<- factor(as.numeric(test$class==cls), levels = c(0,1))
    roc_o<- roc(as.numeric(truth), prob, quiet=TRUE)
    cm_b <- confusionMatrix(pred, truth)
    results_traj_bin <- bind_rows(results_traj_bin,
                                  tibble(window = paste0('0-', d),
                                         class  = factor(cls, levels = class_levels),
                                         AUC = as.numeric(auc(roc_o)),
                                         Accuracy = cm_b$overall['Accuracy'],
                                         Sensitivity = cm_b$byClass['Sensitivity'],
                                         Specificity = cm_b$byClass['Specificity']))
  }
}

save_plot(
  ggplot(results_traj_multi, aes(window, Accuracy, group = 1)) +
    geom_line() + geom_point() + scale_y_continuous(limits = c(0,1)) +
    labs(title = 'Trajectory Multiclass Accuracy', x = 'Window', y = 'Accuracy') + theme_custom,
  'traj_multiclass_accuracy.png'
)
for (m in metrics) {
  save_plot(
    ggplot(results_traj_bin,
           aes(x = window, y = .data[[m]], color = class, group = class)) +
      geom_line() + geom_point() + scale_y_continuous(limits = c(0,1)) +
      scale_color_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste('Trajectory', m, 'One-vs-Rest'), x = 'Window', y = m) + theme_custom,
    paste0('traj_binary_', tolower(m), '.png')
  )
}
results_traj_multi %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'traj_multiclass_metrics.csv'), row.names=FALSE)
results_traj_bin %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'traj_binary_metrics.csv'), row.names=FALSE)

# multiclass importance B
topB <- map_dfr(traj_days,
                ~importance_traj_mc[[as.character(.x)]] %>% mutate(window = paste0('0-', .x)))
save_plot(
  ggplot(topB %>% group_by(window) %>% slice_max(Gain, n = 10),
         aes(window, reorder(Feature, Gain), fill = Gain)) +
    geom_tile() +
    labs(title = 'Trajectory Multiclass Feature Importance',
         x = 'Window', y = 'Feature') + theme_custom,
  'traj_mc_feature_importance.png'
)
# binary importance B
for (cls in levels(results_traj_bin$class)) {
  dfB <- map_dfr(traj_days,
                 ~importance_traj_bin[[as.character(.x)]][[cls]] %>% mutate(window = paste0('0-', .x)))
  save_plot(
    ggplot(dfB %>% group_by(window) %>% slice_max(Gain, n = 10),
           aes(window, reorder(Feature, Gain), fill = Gain)) +
      geom_tile() +
      labs(title = paste('Trajectory Binary Feature Importance -', cls),
           x = 'Window', y = 'Feature') + theme_custom,
    paste0('traj_imp_binary_', cls, '.png')
  )
}

# ----------------------------------------------------------------------------
# Section C: Last Available Value --------------------------------------------
last_days <- 0:7
results_last_multi <- tibble()
results_last_bin   <- tibble()
importance_last_mc <- list()
importance_last_bin<- list()

for (d in last_days) {
  feats <- build_feats_last(d)
  train <- feats[split_idx, ]; test <- feats[-split_idx, ]
  preds <- setdiff(names(train), c('stay_id','class'))
  
  # Multiclass
  dmat  <- xgb.DMatrix(as.matrix(train[, preds]), label = as.numeric(train$class) - 1)
  cv    <- xgb.cv(xgb_multi_params, dmat, nrounds = 300, nfold = 5,
                  early_stopping_rounds = 20, verbose = FALSE)
  mc    <- xgb.train(xgb_multi_params, dmat, nrounds = cv$best_iteration, verbose = FALSE)
  importance_last_mc[[as.character(d)]] <- xgb.importance(preds, model = mc)
  pmat  <- predict(mc, xgb.DMatrix(as.matrix(test[, preds]))) %>%
    matrix(ncol = length(levels(train$class)), byrow = TRUE)
  plab  <- factor(levels(train$class)[max.col(pmat)], levels = levels(train$class))
  cm_mc <- confusionMatrix(plab, test$class)
  results_last_multi <- bind_rows(
    results_last_multi,
    tibble(day = d, Accuracy = cm_mc$overall['Accuracy'])
  )
  
  # Binary
  importance_last_bin[[as.character(d)]] <- list()
  for (cls in levels(train$class)) {
    ytr  <- as.numeric(train$class == cls)
    bmat <- xgb.DMatrix(as.matrix(train[, preds]), label = ytr)
    cvb  <- xgb.cv(xgb_bin_params, bmat, nrounds = 300, nfold = 5,
                   early_stopping_rounds = 20, verbose = FALSE)
    bin  <- xgb.train(xgb_bin_params, bmat, nrounds = cvb$best_iteration, verbose = FALSE)
    importance_last_bin[[as.character(d)]][[cls]] <- xgb.importance(preds, model = bin)
    prob <- predict(bin, xgb.DMatrix(as.matrix(test[, preds])))
    pred <- factor(as.numeric(prob > 0.5), levels = c(0,1))
    truth<- factor(as.numeric(test$class == cls), levels = c(0,1))
    roc_o<- roc(as.numeric(truth), prob, quiet = TRUE)
    cm_b <- confusionMatrix(pred, truth)
    results_last_bin <- bind_rows(results_last_bin,
                                  tibble(
                                    day         = d,
                                    class       = factor(cls, levels = class_levels),
                                    AUC         = as.numeric(auc(roc_o)),
                                    Accuracy    = cm_b$overall['Accuracy'],
                                    Sensitivity = cm_b$byClass['Sensitivity'],
                                    Specificity = cm_b$byClass['Specificity']
                                  ))
  }
}

save_plot(
  ggplot(results_last_multi, aes(day, Accuracy, group = 1)) +
    geom_line() + geom_point() + scale_y_continuous(limits = c(0,1)) +
    labs(title = 'Last-Value Multiclass Accuracy', x = 'Day', y = 'Accuracy') + theme_custom,
  'last_multiclass_accuracy.png'
)
for (m in metrics) {
  save_plot(
    ggplot(results_last_bin,
           aes(x = day, y = .data[[m]], color = class, group = class)) +
      geom_line() + geom_point() + scale_y_continuous(limits = c(0,1)) +
      scale_color_manual(values = class_cols_named, drop = FALSE) +
      labs(title = paste('Last-Value', m, 'One-vs-Rest'), x = 'Day', y = m) + theme_custom,
    paste0('last_binary_', tolower(m), '.png')
  )
}
results_last_multi %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'last_multiclass_metrics.csv'), row.names=FALSE)
results_last_bin %>% mutate(across(where(is.numeric), round, 2)) %>%
  write.csv(file.path(out_dir,'last_binary_metrics.csv'), row.names=FALSE)

# ----------------------------------------------------------------------------
# Extra importance visualisations (ordered heat-map, ridgeline, bump chart) --
imp_dir <- file.path(out_dir, "importance_profiles")
if (!dir.exists(imp_dir)) dir.create(imp_dir)

ordered_heatmap <- function(df, day_var) {
  df %>%
    mutate(
      day_tag  = as.integer(str_extract(Feature, "(?<=day)\\d+")),
      var_base = str_remove(Feature, "^day\\d+_"),
      day_tag  = if_else(is.na(day_tag), -1L, day_tag)
    ) %>%
    arrange(day_tag, var_base) %>%
    mutate(Feature = factor(Feature, levels = unique(Feature)),
           {{day_var}} := !!sym(day_var))
}

plot_heat <- function(df, day_name, file_stub, day_var) {
  save_plot(
    ggplot(df %>% group_by(!!sym(day_var)) %>% slice_max(Gain, n = 10),
           aes(x = !!sym(day_var), y = Feature, fill = Gain)) +
      geom_tile() +
      labs(title = paste(day_name, "Multiclass Feature Importance (ordered)"),
           x = stringr::str_to_title(day_var), y = "Feature") +
      theme_custom,
    paste0(file_stub, "_heatmap_ordered.png")
  )
}

plot_ridge <- function(df, day_name, file_stub, day_var) {
  save_plot(
    ggplot(df, aes(x = !!sym(day_var), y = Feature, height = Gain,
                   group = Feature, fill = Feature)) +
      geom_ridgeline(scale = 0.9, alpha = 0.7, colour = NA) +
      scale_fill_viridis_d(option = "C", guide = "none") +
      labs(title = paste(day_name, "Ridgeline of Gain over Time"),
           x = stringr::str_to_title(day_var), y = "Feature") +
      theme_custom,
    paste0(file_stub, "_ridgeline.png")
  )
}

plot_bump <- function(df, day_name, file_stub, day_var, k = 10) {
  rank_df <- df %>%
    group_by(!!sym(day_var)) %>%
    arrange(desc(Gain)) %>%
    mutate(rank = row_number()) %>%
    filter(rank <= k) %>%
    ungroup()
  save_plot(
    ggplot(rank_df,
           aes(x = !!sym(day_var), y = rank, colour = Feature, group = Feature)) +
      geom_line(size = 1.2) + geom_point(size = 2) +
      scale_y_reverse(breaks = 1:k) +
      labs(title = paste(day_name, "Top", k, "Feature Rank (bump chart)"),
           x = stringr::str_to_title(day_var), y = "Rank (1 = highest)") +
      theme_custom + theme(legend.position = "none"),
    paste0(file_stub, "_bump.png")
  )
}

# ----- Static ---------------------------------------------------------------
static_df <- map_dfr(static_days,
                     ~importance_static_mc[[as.character(.x)]] %>% mutate(day = .x))
static_ord <- ordered_heatmap(static_df, "day")
plot_heat(static_ord, "Static", file.path(imp_dir, "static"), "day")
plot_ridge(static_ord, "Static", file.path(imp_dir, "static"), "day")
plot_bump(static_ord, "Static", file.path(imp_dir, "static"), "day")

# ----- Trajectory -----------------------------------------------------------
traj_df <- map_dfr(traj_days,
                   ~importance_traj_mc[[as.character(.x)]] %>%
                     mutate(window = paste0("0-", .x)))
traj_ord <- ordered_heatmap(traj_df, "window")
plot_heat(traj_ord, "Trajectory", file.path(imp_dir, "traj"), "window")
plot_ridge(traj_ord, "Trajectory", file.path(imp_dir, "traj"), "window")
plot_bump(traj_ord, "Trajectory", file.path(imp_dir, "traj"), "window")

# ----- Last-value -----------------------------------------------------------
last_df <- map_dfr(last_days,
                   ~importance_last_mc[[as.character(.x)]] %>% mutate(day = .x))
last_ord <- ordered_heatmap(last_df, "day")
plot_heat(last_ord, "Last-Value", file.path(imp_dir, "last"), "day")
plot_ridge(last_ord, "Last-Value", file.path(imp_dir, "last"), "day")
plot_bump(last_ord, "Last-Value", file.path(imp_dir, "last"), "day")

# ---------------------------------------------------------------------------
# Variable–profile plots (days 0-14) with n≥10 filter ------------------------
prof_out <- file.path(out_dir, "var_profiles")
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
    theme_custom
  
  ggsave(file.path(prof_out, paste0(v, " 28day.png")), p, width = 9, height = 6)
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


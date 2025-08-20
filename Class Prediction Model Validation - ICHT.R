# ---------------------------------------------------------------------------
# static_mimic_to_icht_d0-7.R  –  cumulative-day static (all-values) models
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr)
  library(xgboost); library(caret); library(pROC); library(ggplot2)
})

out_dir <- "Class prediction validation v2"
dir.create(out_dir, showWarnings = FALSE)

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

save_reliability <- function(cal_df, title, file_stub) {
  p <- cal_df %>%
    filter(bin_count > 0) %>%
    ggplot(aes(exp, obs, size = bin_count)) +
    geom_point(alpha = .7) +
    geom_abline(linetype = "dashed") +
    scale_size_continuous(range = c(1, 6)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = title, x = "Predicted", y = "Observed", size = "n") +
    theme_bw()
  ggsave(filename = file.path(out_dir, file_stub), plot = p, width = 5, height = 5)
}

# ---------------- 1.  Common objects ---------------------------------------
measure <- c("norad_vasorate","avg_lactate","locf_bicarbonate",
             "avg_peak_insp_pressure","avg_peep","avg_pao2fio2ratio",
             "locf_creatinine","avg_pco2","avg_minute_volume","avg_resp_rate")

# Updated static/all-values hyperparameters (multiclass & binary)
param_mc  <- list(booster="gbtree", objective="multi:softprob",
                  eval_metric="mlogloss", eta=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=1.0)
param_bin <- list(booster="gbtree", objective="binary:logistic",
                  eval_metric="auc", eta=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=1.0)

# ---------------- 2.  MIMIC long → day-wide builder ------------------------
prob_mim  <- read.csv("Data/pprob_MIMIC.csv") %>% select(stay_id, class)
long_mim <- read.csv("Data/mimic 3d cohort 28 days var.csv") %>%
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
prob_icht <- read.csv("Data/pprob_ICHT.csv") %>% select(stay_id, class)
long_icht <- read.csv("Data/ICHT 28day var for class prediction.csv") %>%
  select(-c(avg_pco2, avg_pf_ratio)) %>%
  mutate(
    age        = as.numeric(age),
    gender_bin = if_else(gender %in% c("M", "Male", 1), 1, 0)
  )

static_icht <- long_icht %>% select(stay_id, age, gender_bin) %>% distinct()

name_map <- c(
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
  rename(!!!name_map) %>%
  mutate(avg_pao2fio2ratio = ifelse(avg_pao2fio2ratio > 470, NA, avg_pao2fio2ratio))

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

# ---------------- 4.  Storage tibbles --------------------------------------
frozen_mc  <- tibble()
frozen_bin <- tibble()
recal_mc   <- tibble()
recal_bin  <- tibble()
frozen_mc_mimic <- tibble()

# ---------------- 5.  Loop over d = 0 … 7 ----------------------------------
for (d in 0:7){
  message(">>>>  Day window 0-", d, "  <<<<")
  
  # ----- 5A  build matrices -------------------------------------------------
  mim  <- wide_mim(d)
  icht <- wide_icht(d)
  preds<- setdiff(names(mim), c("stay_id","class"))
  
  # ----- 5B  MIMIC train (to freeze) ---------------------------------------
  set.seed(1 + d)                                   # seed per day
  idx_m <- createDataPartition(mim$class, p = .8, list = FALSE)
  tr_m  <- mim[idx_m,];  te_m <- mim[-idx_m,]      # te_m not used for metrics here
  
  param_mc_m <- c(param_mc, list(num_class = length(levels(tr_m$class))))
  
  # Multiclass MIMIC (frozen)
  dmat_m_mc <- xgb.DMatrix(as.matrix(tr_m[, preds]), label = as.numeric(tr_m$class) - 1)
  cv_m_mc   <- xgb.cv(param_mc_m, dmat_m_mc, nrounds = 300, nfold = 5,
                      early_stopping_rounds = 20, verbose = 0)
  mc_m      <- xgb.train(param_mc_m, dmat_m_mc, nrounds = cv_m_mc$best_iteration, verbose = 0)
  
  # ---- Frozen model evaluated on MIMIC hold-out (for comparison) ------------
  pmat_mimic <- predict(mc_m, xgb.DMatrix(as.matrix(te_m[, preds]))) %>%
    matrix(ncol = length(class_levels), byrow = TRUE)
  acc_mimic  <- confusionMatrix(
    factor(class_levels[max.col(pmat_mimic)], levels = class_levels),
    te_m$class
  )$overall["Accuracy"]
  frozen_mc_mimic <- bind_rows(frozen_mc_mimic, tibble(day = d, Accuracy = acc_mimic))
  
  
  # Binary OvR MIMIC (frozen)
  bin_m <- map(setNames(levels(mim$class), levels(mim$class)), \(cls){
    y <- as.numeric(tr_m$class == cls)
    dmat <- xgb.DMatrix(as.matrix(tr_m[, preds]), label = y)
    cvb  <- xgb.cv(param_bin, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
    xgb.train(param_bin, dmat, nrounds = cvb$best_iteration, verbose = 0)
  })
  
  # ----- 5C  Split ICHT once; use same hold-out for BOTH frozen & recal ----
  set.seed(2 + d)
  idx_i <- createDataPartition(icht$class, p = .8, list = FALSE)
  tr_i  <- icht[idx_i,];  te_i <- icht[-idx_i,]
  
  # ===================  FROZEN (MIMIC → ICHT test)  ========================
  # Multiclass
  pmat_frozen <- predict(mc_m, xgb.DMatrix(as.matrix(te_i[, preds]))) %>%
    matrix(ncol = length(class_levels), byrow = TRUE)
  plab_frozen <- factor(class_levels[max.col(pmat_frozen)], levels = class_levels)
  cm_mc_f     <- confusionMatrix(plab_frozen, te_i$class)
  # Multiclass calibration (pooled ECE; multiclass Brier via one-hot)
  Y_onehot_f  <- model.matrix(~ class - 1, data = te_i)
  col_order   <- paste0("class", class_levels)
  if (!all(colnames(Y_onehot_f) == col_order)) Y_onehot_f <- Y_onehot_f[, col_order, drop = FALSE]
  brier_mc_f  <- mean((pmat_frozen - Y_onehot_f)^2)
  cal_mc_f    <- ece_brier(as.numeric(pmat_frozen), as.numeric(Y_onehot_f))
  save_reliability(cal_mc_f$df,
                   sprintf("Frozen multiclass Day %d", d),
                   sprintf("cal_frozen_multiclass_day%02d.png", d))
  frozen_mc <- bind_rows(frozen_mc,
                         tibble(day = d, Accuracy = cm_mc_f$overall["Accuracy"],
                                Brier = brier_mc_f, ECE = cal_mc_f$ece))
  
  # Binary
  frozen_bin <- bind_rows(
    frozen_bin,
    map_dfr(class_levels, \(cls){
      prob  <- predict(bin_m[[cls]], xgb.DMatrix(as.matrix(te_i[, preds])))
      yte   <- as.numeric(te_i$class == cls)
      truth <- factor(yte, levels = c(0,1))
      pred  <- factor(as.numeric(prob > .5), levels = c(0,1))
      roc_o <- roc(as.numeric(truth), prob, quiet = TRUE)
      cm_b  <- confusionMatrix(pred, truth)
      cal_b <- ece_brier(prob, yte)
      save_reliability(cal_b$df,
                       sprintf("Frozen %s Day %d", cls, d),
                       sprintf("cal_frozen_%s_day%02d.png", cls, d))
      tibble(day = d, class = cls,
             AUC = as.numeric(auc(roc_o)),
             Accuracy = cm_b$overall["Accuracy"],
             Sensitivity = cm_b$byClass["Sensitivity"],
             Specificity = cm_b$byClass["Specificity"],
             Brier = cal_b$brier, ECE = cal_b$ece)
    })
  )
  
  # ===================  RECALIBRATED (train ICHT → test ICHT)  =============
  # Multiclass
  param_mc_i <- c(param_mc, list(num_class = length(levels(tr_i$class))))
  dmat_i_mc  <- xgb.DMatrix(as.matrix(tr_i[, preds]), label = as.numeric(tr_i$class) - 1)
  cv_i_mc    <- xgb.cv(param_mc_i, dmat_i_mc, 300, 5, early_stopping_rounds = 20, verbose = 0)
  mc_i       <- xgb.train(param_mc_i, dmat_i_mc, nrounds = cv_i_mc$best_iteration, verbose = 0)
  
  pmat_recal <- predict(mc_i, xgb.DMatrix(as.matrix(te_i[, preds]))) %>%
    matrix(ncol = length(class_levels), byrow = TRUE)
  plab_recal <- factor(class_levels[max.col(pmat_recal)], levels = class_levels)
  cm_mc_r    <- confusionMatrix(plab_recal, te_i$class)
  # Multiclass calibration
  Y_onehot_r <- model.matrix(~ class - 1, data = te_i)
  if (!all(colnames(Y_onehot_r) == col_order)) Y_onehot_r <- Y_onehot_r[, col_order, drop = FALSE]
  brier_mc_r <- mean((pmat_recal - Y_onehot_r)^2)
  cal_mc_r   <- ece_brier(as.numeric(pmat_recal), as.numeric(Y_onehot_r))
  save_reliability(cal_mc_r$df,
                   sprintf("Recal multiclass Day %d", d),
                   sprintf("cal_recal_multiclass_day%02d.png", d))
  recal_mc <- bind_rows(recal_mc,
                        tibble(day = d, Accuracy = cm_mc_r$overall["Accuracy"],
                               Brier = brier_mc_r, ECE = cal_mc_r$ece))
  
  # Binary
  recal_bin <- bind_rows(
    recal_bin,
    map_dfr(class_levels, \(cls){
      ytr  <- as.numeric(tr_i$class == cls)
      dmat <- xgb.DMatrix(as.matrix(tr_i[, preds]), label = ytr)
      cvb  <- xgb.cv(param_bin, dmat, 300, 5, early_stopping_rounds = 20, verbose = 0)
      bst  <- xgb.train(param_bin, dmat, nrounds = cvb$best_iteration, verbose = 0)
      prob  <- predict(bst, xgb.DMatrix(as.matrix(te_i[, preds])))
      yte   <- as.numeric(te_i$class == cls)
      truth <- factor(yte, levels = c(0,1))
      pred  <- factor(as.numeric(prob > .5), levels = c(0,1))
      roc_o <- roc(as.numeric(truth), prob, quiet = TRUE)
      cm_b  <- confusionMatrix(pred, truth)
      cal_b <- ece_brier(prob, yte)
      save_reliability(cal_b$df,
                       sprintf("Recal %s Day %d", cls, d),
                       sprintf("cal_recal_%s_day%02d.png", cls, d))
      tibble(day = d, class = cls,
             AUC = as.numeric(auc(roc_o)),
             Accuracy = cm_b$overall["Accuracy"],
             Sensitivity = cm_b$byClass["Sensitivity"],
             Specificity = cm_b$byClass["Specificity"],
             Brier = cal_b$brier, ECE = cal_b$ece)
    })
  )
}

# ---------------- 6.  write CSV + quick plots ------------------------------
write.csv(frozen_mc , file.path(out_dir,"frozen_multiclass_byDay.csv"), row.names=FALSE)
write.csv(frozen_bin, file.path(out_dir,"frozen_binary_byDay.csv")   , row.names=FALSE)
write.csv(recal_mc  , file.path(out_dir,"recal_multiclass_byDay.csv"), row.names=FALSE)
write.csv(recal_bin , file.path(out_dir,"recal_binary_byDay.csv")    , row.names=FALSE)

# Multiclass accuracy (frozen vs recalibrated)
ggsave(file.path(out_dir,"multiclass_accuracy_frozen_vs_recal.png"),
       ggplot() +
         geom_line(data=frozen_mc,aes(day,Accuracy,colour="Frozen")) +
         geom_line(data=recal_mc ,aes(day,Accuracy,colour="Recal")) +
         scale_colour_manual(values=c(Frozen="steelblue",Recal="darkred")) +
         ylim(0,1)+labs(colour=NULL,y="Accuracy",x="Last day included (0–d)")+
         theme_bw(), width=6,height=4)

# Optional: Brier/ECE comparisons for multiclass
for (m in c("Brier","ECE")) {
  p <- ggplot() +
    geom_line(data=frozen_mc, aes(day, .data[[m]], colour="Frozen")) +
    geom_line(data=recal_mc , aes(day, .data[[m]], colour="Recal")) +
    scale_colour_manual(values=c(Frozen="steelblue", Recal="darkred")) +
    labs(colour=NULL, y=m, x="Last day included (0–d)") + theme_bw()
  ggsave(file.path(out_dir, paste0("multiclass_", tolower(m), "_frozen_vs_recal.png")),
         p, width=6, height=4)
}

# ---------------------------------------------------------------------------
# 7.  Curves for binary models – AUC / Acc / Sens / Spec / Brier / ECE
# ---------------------------------------------------------------------------
metrics <- c("AUC", "Accuracy", "Sensitivity", "Specificity", "Brier", "ECE")
for (m in metrics) {
  p <- ggplot() +
    geom_line(data = frozen_bin,
              aes(x = day, y = .data[[m]], colour = class, linetype = "Frozen")) +
    geom_line(data = recal_bin,
              aes(x = day, y = .data[[m]], colour = class, linetype = "Recal")) +
    scale_colour_brewer(palette = "Dark2") +
    scale_linetype_manual(values = c(Frozen = "solid", Recal = "dashed")) +
    labs(title = paste(m, "over days – binary one-vs-rest"),
         x = "Last day included (0–d)", y = m,
         colour = "Class", linetype = NULL) +
    theme_bw()
  ggsave(file.path(out_dir, paste0("binary_", tolower(m), "_frozen_vs_recal.png")),
         p, width = 7, height = 4)
}

# Multiclass accuracy: Frozen (ICHT) vs Recal (ICHT) vs Frozen (MIMIC)
ggsave(file.path(out_dir,"multiclass_accuracy_frozenICHT_recalICHT_frozenMIMIC.png"),
       ggplot() +
         geom_line(data = frozen_mc,        aes(day, Accuracy, colour = "Frozen on ICHT")) +
         geom_line(data = recal_mc,         aes(day, Accuracy, colour = "Recal on ICHT")) +
         geom_line(data = frozen_mc_mimic,  aes(day, Accuracy, colour = "Frozen on MIMIC")) +
         scale_colour_manual(values = c("Frozen on ICHT" = "steelblue",
                                        "Recal on ICHT"  = "darkred",
                                        "Frozen on MIMIC"= "darkgreen")) +
         ylim(0, 1) +
         labs(colour = NULL, y = "Accuracy", x = "Last day included (0–d)") +
         theme_bw(),
       width = 6, height = 4)

message("✅  All results + calibration plots saved to: ", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 9 · DAILY VARIABLE TRAJECTORIES · ICHT cohort, K = 4 classes
#     (mean ± SE, days 0-14, legend at bottom, x-breaks 0–14 by 2)
# ─────────────────────────────────────────────────────────────────────────────
prof_out_icht <- file.path(out_dir, "var_profiles_ICHT")
if (!dir.exists(prof_out_icht)) dir.create(prof_out_icht)

var_long_icht <- long_icht %>%                              # uses long_icht from Step 3
  filter(days_from_start <= 14) %>%
  select(stay_id, day_period = days_from_start, all_of(measure)) %>%
  pivot_longer(-c(stay_id, day_period),
               names_to = "variable", values_to = "value") %>%
  left_join(prob_icht %>%                                   # class labels
              mutate(class = factor(paste0("C", class), levels = class_levels)),
            by = "stay_id")

for (v in measure) {
  tab <- var_long_icht %>%
    filter(variable == v) %>%
    group_by(class, day_period) %>%
    summarise(
      n_total = n(),
      n_miss  = sum(is.na(value)),
      n_keep  = n_total - n_miss,
      p_miss  = n_miss / n_total,
      mean_v  = mean(value, na.rm = TRUE),
      se_v    = sd(value,  na.rm = TRUE) / sqrt(n_keep),
      .groups = "drop"
    ) %>%
    mutate(
      mean_v = if_else(n_keep < 10, NA_real_, mean_v),
      se_v   = if_else(n_keep < 10, NA_real_, se_v),
      p_miss = if_else(n_keep < 10, NA_real_, p_miss)
    ) %>%
    filter(!is.na(mean_v))
  
  if (nrow(tab) == 0) next
  
  p <- ggplot(tab, aes(day_period, mean_v,
                       colour = class, group = class)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean_v - se_v,
                    ymax = mean_v + se_v,
                    fill = class),
                alpha = 0.15, colour = NA) +
    geom_point(aes(size = p_miss), alpha = 0.7) +
    scale_color_manual(values = class_cols_named, drop = FALSE) +
    scale_fill_manual(values = class_cols_named, guide = "none", drop = FALSE) +
    scale_size_continuous(
      range  = c(0.5, 4),
      breaks = c(0, 0.25, 0.5, 0.75),
      labels = scales::percent_format(accuracy = 1)
    ) +
    scale_x_continuous(breaks = seq(0, 14, 2), limits = c(0, 14)) +
    labs(title = v,
         x = "ICU day",
         y = "Mean ± SE",
         size = "% missing") +
    theme_bw() +
    theme(legend.position = "bottom",
          plot.title      = element_text(hjust = .5, face = "bold"))
  
  ggsave(file.path(prof_out_icht, paste0(v, "_14day_ICHT.png")), p,
         width = 9, height = 6, dpi = 300)
}


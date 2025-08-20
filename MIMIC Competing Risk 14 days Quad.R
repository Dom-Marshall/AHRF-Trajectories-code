# ===============================================================
#   Joint latent-class mixed model (JLMM) with competing risks
#   MIMIC-IV 
# ===============================================================

# ===============================================================
#   JLMM with competing risks  –  MIMIC-IV  (v2)
#   *expects:  mimic_pf_daily.csv  &  mimic_stays.csv*
# ===============================================================

# -------- CONFIG -----------------------------------------------------------
cfg <- list(
  csv_pf           = "mimic_pf_daily.csv",
  csv_stay         = "mimic_stays.csv",
  out_dir          = "Results_MIMIC_multiclass_jm_full_14d2",
  admin_day        = 14,
  k_range          = 2:2,
  rep_starts       = 20,
  max_iter         = 200,
  parallel_type    = "PSOCK",
  seed_global      = 123
)
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(lcmm)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(parallel)
  library(doParallel)
  library(foreach)
  library(future)
  library(survival)
  library(stringr)
})

dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(cfg$seed_global)

# -------- 1. LOAD & HARMONISE ----------------------------------------------
pf_daily <- read.csv(cfg$csv_pf)        # ← has days_from_start already
stays    <- read.csv(cfg$csv_stay)      # ← stay_id, timezero, outtime, hospital_expire_flag

parse_datetime <- function(datetime_str) {
  datetime_str <- as.character(datetime_str)
  datetime_str <- trimws(datetime_str)
  datetime_str[datetime_str == ""] <- NA
  suppressWarnings(as.POSIXct(datetime_str, format = "%d/%m/%Y %H:%M", tz = "UTC"))
}

stays$timezero <- parse_datetime(stays$timezero)
stays$outtime  <- parse_datetime(stays$outtime )


# 1.1 merge & rename so downstream code is 100 % MIMIC-compatible -------------
raw <- pf_daily %>%
  mutate(
    stay_id      = stay_id,                       # keep same name
    day_period   = days_from_start,               # already integer 0,1,2…
    pf_ratio_avg = avg_pf_ratio
  ) %>%
  select(stay_id, day_period, pf_ratio_avg)

raw <- raw %>%
  left_join(stays %>% 
              select(stay_id, timezero, outtime, icu_mort),
            by = "stay_id") %>%
  arrange(stay_id, day_period)

# -------- 2.  BUILD COMPETING-RISK STATUS TABLE ----------------------------
cr_tab <- raw %>%
  distinct(stay_id, timezero, outtime, icu_mort) %>%
  mutate(
    deathtime = ifelse(icu_mort == 1, outtime, NA),
    # raw time-to-event in days
    raw_tte   = as.numeric(difftime(coalesce(outtime, timezero),
                                    timezero,
                                    units = "days")),
    status = case_when(            # 1 = death, 2 = discharge, 0 = censored
      icu_mort == 1 ~ 1L,
      !is.na(outtime)           ~ 2L,
      TRUE                      ~ 0L
    ),
    event = if_else(raw_tte > cfg$admin_day, 0L, status),
    tte   = pmin(raw_tte, cfg$admin_day)
  ) %>% 
  select(stay_id, tte, event)

# -------- 2.1  SANITY-CHECK COUNTS BY DAY 14 -------------------------------
check_tab <- cr_tab %>%
  mutate(event_lbl = factor(event,
                            levels = c(0,1,2),
                            labels = c("Censored @ 14 d",
                                       "Death ≤ 14 d",
                                       "Discharged ≤ 14 d"))) %>%
  count(event_lbl, name = "N")

print(check_tab)
write.csv(check_tab, file.path(cfg$out_dir, "SanityCheck_D14_counts.csv"),
          row.names = FALSE)

# -------- 3.  BUILD JOINT-MODEL DATA FRAME ---------------------------------
make_joint <- function(df) {
  df %>% 
    left_join(cr_tab, by = "stay_id") %>%
    filter(day_period <= floor(tte)) %>%
    mutate(event = as.integer(event))
}
joint_data <- as.data.frame(make_joint(raw))

# -------- 4.  FIT MODELS (unchanged) ---------------------------------------
message("→ baseline ng = 1")
base_jm <- Jointlcmm(
  fixed      = pf_ratio_avg ~ day_period + I(day_period^2),
  random     = ~ day_period,
  subject    = "stay_id",
  survival   = Surv(tte, event) ~ 1,
  hazard     = c("Weibull","Weibull"),
  hazardtype = "Specific",
  link       = "linear",
  ng         = 1,
  data       = joint_data,
  na.action  = 1,
  maxiter    = cfg$max_iter
)
saveRDS(base_jm, file.path(cfg$out_dir, "mimic_K1_competing.rds"))

# 4.1 grid-search for K = 2:5  (identical to MIMIC) --------------------------
cores <- future::availableCores()
cl <- if (cfg$parallel_type == "FORK") {
  makeCluster(cores, type = "FORK")
} else {
  makeCluster(max(1, cores - 2), type = "PSOCK")
}
clusterEvalQ(cl, { library(lcmm); library(survival); library(tidyr) })
clusterSetRNGStream(cl, iseed = 2025)  
joint_models <- vector("list", max(cfg$k_range))
joint_models[[1]] <- base_jm

for (ng in cfg$k_range) {
  message("→ grid-search ng = ", ng)
  clusterExport(cl, "ng", envir = environment())
  joint_models[[ng]] <- tryCatch({
    gridsearch(
      rep     = cfg$rep_starts,
      maxiter = cfg$max_iter,
      minit   = base_jm,
      Jointlcmm(
        fixed      = pf_ratio_avg ~ day_period + I(day_period^2),
        mixture    = pf_ratio_avg ~ day_period + I(day_period^2),
        random     = ~ day_period,
        subject    = "stay_id",
        survival   = Surv(tte, event) ~ 1,
        hazard     = c("Weibull","Weibull"),
        hazardtype = "Specific",
        link       = "linear",
        ng         = ng,
        data       = joint_data,
        na.action  = 1
      ), cl = cl)
  }, error = function(e) { message("gridsearch failed (ng=",ng,"):",e$message); NULL })
}
stopCluster(cl)
saveRDS(joint_models, file.path(cfg$out_dir, "mimic_multiclass_joint_models_full_14d.rds"))

# -------- 5.  DOWNSTREAM PLOTS, TABLES, CIFs -------------------------------

# ---- Post-Fit Summary ------------------------------------------------------
converged <- Filter(Negate(is.null), joint_models)

fit_stats <- map_dfr(converged, function(m) {
  bic     <- m$BIC
  postcols<- grep("^prob", names(m$pprob), value = TRUE)
  post    <- as.matrix(m$pprob[, postcols])
  post[post < 1e-16] <- 1e-16
  
  N <- nrow(post)
  G <- ncol(post)
  
  # 1) Shannon entropy H
  H <- -sum(post * log(post))
  
  # 2) two normalized‐entropy measures
  entropy_norm1 <- if (G==1) 1 else   H         / (N * log(G))
  entropy_norm2 <- if (G==1) 1 else 1 - H         / (N * log(G))
  
  # 3) ICL1' = BIC - H;   ICL1 = BIC - 2H
  ICL1p <- bic -       H
  ICL1  <- bic - 2 *   H
  
  # 4) ICL2 = BIC - 2 sum(log(max prob))
  assigned <- max.col(post)
  lp       <- log(post[cbind(seq_len(N), assigned)])
  ICL2     <- bic - 2 * sum(lp)
  
  # 5) minimum class size
  class_sizes <- table(factor(assigned, levels = 1:G))
  min_class   <- min(class_sizes)
  
  tibble(
    K              = m$ng,
    logLik         = m$loglik,
    BIC            = bic,
    H              = H,
    Entropy_norm1  = entropy_norm1,
    Entropy_norm2  = entropy_norm2,
    ICL1p          = ICL1p,
    ICL1           = ICL1,
    ICL2           = ICL2,
    MinClassSize   = as.integer(min_class)
  )
}) %>%
  arrange(K)

fit_stats
# --- 2) Pivot longer over all four metrics ---
df_long <- fit_stats %>% select(K, BIC, ICL1, Entropy_norm2, MinClassSize) %>%
  pivot_longer(
    cols      = c("BIC", "ICL1", "Entropy_norm2", "MinClassSize"),
    names_to  = "Metric",
    values_to = "Value"
  )

# --- 3) Plot ---
Kmax <- max(fit_stats$K)  

p <- ggplot(df_long, aes(x = K, y = Value, color = Metric)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Metric, scales = "free_y", ncol = 4) +
  scale_x_continuous(
    breaks      = 1:Kmax,
    minor_breaks = NULL
  ) +
  labs(
    title = "Model fit metrics",
    x     = "Number of classes (K)",
    y     = NULL
  ) +
  theme_minimal() +
  theme(
    panel.border     = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position  = "none"
  )


ggsave(
  filename = file.path(cfg$out_dir, sprintf("CR_14d_SummaryPlot.pdf")),
  plot = p, width = 10, height =6
)

write.csv(fit_stats, file.path(cfg$out_dir, "SummaryTable_MIMIC_CR_14d.csv"), row.names = FALSE)

# ---- Helpers for class mean curves (uses the model's parameterisation) ----
normalize_class_levels <- function(g, ng) {
  factor(paste0("Class ", g), levels = paste0("Class ", seq_len(ng)))
}
coef_names <- c("intercept","day_period","I(day_period^2)")

get_beta_g <- function(pars, g){
  idx <- vapply(coef_names, function(nm){
    pos <- which(names(pars) == paste0(nm, " class", g))
    if (!length(pos)) stop("coef not found: ", nm, " class", g)
    tail(pos, 1L)  # intercept can appear twice; take latent-process one
  }, integer(1))
  out <- pars[idx]; names(out) <- coef_names; out
}

class_mean_curve_safe <- function(mdl, times){
  p  <- mdl$best
  a0 <- p["Linear 1"]; a1 <- p["Linear 2"]
  X  <- cbind(1, times, times^2); colnames(X) <- coef_names
  dplyr::bind_rows(lapply(seq_len(mdl$ng), function(g){
    b <- get_beta_g(p, g)
    yhat <- as.numeric(a0 + a1 * (X %*% b))
    tibble::tibble(Class = normalize_class_levels(g, mdl$ng),
                   day_period = times,
                   value = yhat,
                   type  = "Model predicted")
  }))
}

# ---- Trajectory Plots ------------------------------------------------------
time_seq    <- seq(0, cfg$admin_day, by = 1)
all_colours <- c("deeppink","deepskyblue","forestgreen","orangered","darkorchid")

# latent-process covariates (must match Jointlcmm fixed/mixture)
coef_names <- c("intercept", "day_period", "I(day_period^2)")

# a design matrix for μ*(t) = β0 + β1·t + β2·t^2
design_df  <- data.frame(
  intercept         = 1,
  day_period        = time_seq,
  `I(day_period^2)` = time_seq^2,
  check.names       = FALSE
)
design_mat <- as.matrix(design_df[, coef_names, drop = FALSE])

# axis breaks
x_breaks <- seq(0, cfg$admin_day, by = 4)
y_breaks <- c(50, 100, 150, 200, 250, 300, 350)

# ---- Loop over models ----
for (ng in seq_len(5)) {
  tryCatch({
    mdl <- joint_models[[ng]]
    if (is.null(mdl)) {
      message(sprintf("Skipping ng=%d: model is NULL", ng))
      next
    }
    
    if (ng == 1) {
      # — ng = 1: use predictY() with correct column names —
      pred <- predictY(
        mdl,
        newdata  = data.frame(day_period = time_seq),
        var.time = "day_period",
        draws    = TRUE
      )
      df <- data.frame(
        day_period = as.numeric(pred$times),
        Ypred      = pred$pred[, "50_class1"],
        lower      = pred$pred[, "2.5_class1"],
        upper      = pred$pred[, "97.5_class1"]
      )
      
      p <- ggplot(df, aes(x = day_period, y = Ypred)) +
        geom_ribbon(aes(ymin = lower, ymax = upper),
                    fill = all_colours[1], alpha = 0.2) +
        geom_line(color = all_colours[1], size = 1) +
        labs(
          title = sprintf("MIMIC JM PF Trajectories (quad) (ng=%d)", ng),
          x     = "Days from Start",
          y     = "PF Ratio"
        ) +
        scale_x_continuous(breaks = x_breaks, limits = c(0, cfg$admin_day)) +
        scale_y_continuous(breaks = y_breaks, limits = c(0, 350)) +
        theme_minimal()
      
    } else {
      # — ng > 1: use class_mean_curve_safe() and keep var names —
      pred_df <- class_mean_curve_safe(mdl, time_seq)
      traj_df <- pred_df %>%
        dplyr::transmute(
          day_period,
          PF_ratio = value,
          Class
        )
      
      p <- ggplot(traj_df, aes(x = day_period, y = PF_ratio, color = Class)) +
        geom_line(size = 1) +
        scale_color_manual(values = all_colours[1:ng]) +
        labs(
          title = sprintf("MIMIC CR PF Trajectories (quad) (ng=%d)", ng),
          x     = "Days from Start",
          y     = "PF Ratio"
        ) +
        scale_x_continuous(breaks = x_breaks, limits = c(0, cfg$admin_day)) +
        scale_y_continuous(breaks = y_breaks, limits = c(0, 350)) +
        theme_minimal()
    }
    
    # save to PDF 
    fname <- file.path(cfg$out_dir,
                       sprintf("pf_trajectories_quad_ng%d.pdf", ng))
    ggsave(filename = fname, plot = p, width = 10, height = 7)
    
  }, error = function(e) {
    message(sprintf("⚠️ Plot failed for ng=%d: %s", ng, e$message))
  })
}


# ---- CIF Curves ------------------------------------------------------------
time_seq    <- seq(1, cfg$admin_day, by = 1)     # for cuminc()
all_colours <- c("deeppink","deepskyblue","forestgreen","orangered","darkorchid")

x_breaks <- seq(0, cfg$admin_day, by = 4)
y_breaks <- seq(0, 1, by = 0.2)

#--- Loop over 1:5 -----------------------------------

for (ng in seq_len(5)) {
  tryCatch({
    mdl <- joint_models[[ng]]
    if (is.null(mdl)) next
    
    # 1) get the full CIF matrix (events 1 & 2) for t=1:cfg$admin_day
    ci_all <- as.data.frame(lcmm::cuminc(
      mdl,
      time      = time_seq,
      draws     = FALSE
    )[[1]])
    
    # 2) prepend t=0 with zeros for all classes
    zero_row <- data.frame(
      event = rep(1:2, each = 1),
      time  = 0,
      as.data.frame(matrix(
        0, nrow = 2, ncol = ng,
        dimnames = list(NULL, paste0("class", 1:ng))
      ))
    )
    ci_df <- bind_rows(zero_row, ci_all)
    
    # 3) pivot longer *all* class columns and compute Surv = 1 - CIF

    surv_df <- ci_df %>%
      mutate(
        Event = factor(event, levels = 1:2,
                       labels = c("Death","Discharge"))
      ) %>%
      pivot_longer(
        cols      = starts_with("class"),
        names_to  = "Class",
        values_to = "CIF"
      ) %>%
      mutate(
        Class    = factor(Class,
                          levels = paste0("class", 1:ng),
                          labels = paste0("Class ", 1:ng)),
        Survival = 1 - CIF,
        Y        = ifelse(Event=="Death", Survival, CIF)
      )
    
    p <- ggplot(surv_df, aes(x = time, y = Y,
                             color     = Class,
                             linetype = Event)) +
      geom_line(size = 1.2) +
      scale_color_manual(values = all_colours[1:ng]) +
      scale_linetype_manual(values = c(Death = "solid",
                                       Discharge = "dashed")) +
      scale_x_continuous(breaks = seq(0,cfg$admin_day,4), limits = c(0,cfg$admin_day)) +
      scale_y_continuous(breaks = seq(0,1,0.2), limits = c(0,1)) +
      labs(
        title = sprintf("MIMIC AHRF Predicted Survival & Incidence (Classes = %d)", ng),
        x     = "Days from start",
        y     = "Survival / Discharge Incidence"
      ) +
      theme_minimal() +
      theme(
        panel.border     = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA)
      )
    
    
    ggsave(
      file.path(cfg$out_dir, sprintf("survival_ng%d.pdf", ng)),
      plot   = p, width = 10, height = 7
    )
  }, error = function(e) {
    message(sprintf("Plot failed for ng=%d: %s", ng, e$message))
  })
}


message("MIMIC pipeline complete – outputs in ", cfg$out_dir)

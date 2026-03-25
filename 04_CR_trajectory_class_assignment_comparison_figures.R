# ===============================================================
#   Use final CRLCMM Model trained on MIMIC
#   *expects: mimic_multiclass_joint_models_full_14d.rds, mimic_pf_daily.csv  &  mimic_stays.csv*
# ===============================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(purrr); library(lcmm); library(cmprsk)
})

# ── 0) CONFIG & PATHS ─────────────────────────────────────────────────────────
if (!exists("cfg")) {
  cfg <- list(
    admin_days = 14,
    out_dir    = "Updated Class comparison figures",
    models = list(MIMIC_mmHg = list(
      rds = "cr_models/mimic_multiclass_joint_models_full_14d.rds",
      K   = 4  #### Set to 4 as this is the model selected 
    )),
    datasets = list(
      MIMIC = list(
        pf   = "Data/mimic_pf_daily.csv",  ##mmHg
        stay = "Data/mimic_stays.csv"
      ),
      ICHT = list(
        pf   = "Data/icht_pf_daily.csv",
        stay = "Data/icht_stays.csv"
      ),
      AUMC = list(
        pf   = "Data/umc_pf_daily.csv",
        stay = "Data/umc_stays.csv"
      )
    )
  )
}
`%||%` <- function(a,b) if (!is.null(a)) a else b

admin    <- if (!is.null(cfg$admin_days)) cfg$admin_days else cfg$admin_day
stopifnot(!is.null(admin))
dir.create(cfg$out_dir %||% ".", showWarnings = FALSE, recursive = TRUE)

x_breaks      <- seq(0, admin, by = 2)
y_breaks_pf   <- c(50, 100, 150, 200, 250, 300, 350, 400)
time_seq      <- 0:admin
time_seq_cif  <- seq(1, admin, by = 1)

# One palette used everywhere
all_colours      <- c("deeppink","deepskyblue","forestgreen","orangered","darkorchid")
class_levels     <- paste0("Class ", 1:4)
class_cols_named <- setNames(all_colours[1:4], class_levels)

# Helper: standardise any class coding to "Class 1"..."Class K"
normalize_class_levels <- function(x, K = 4) {
  x_chr <- trimws(as.character(x))
  lab   <- sub(".*?(\\d+)$", "\\1", x_chr, perl = TRUE)
  lab[!grepl("^\\d+$", lab)] <- NA
  factor(paste("Class", lab), levels = paste("Class", 1:K))
}

# Make axis labels and tick labels larger for all figures
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(purrr); library(lcmm); library(cmprsk)
  library(grid)  # for unit()
})

# Global text sizing (applies to all subsequent plots)

theme_set(theme_minimal(base_size = 14))  # bumps all text baselines
theme_update(
  axis.title        = element_text(size = 18),  # ← AXIS LABELS size 18
  axis.text         = element_text(size = 14),
  axis.ticks        = element_line(linewidth = 0.6),
  axis.ticks.length = unit(6, "pt")
)
# ── 1) LOAD MODEL (K=4) & POSTERIORS ─────────────────────────────────────────
mimic_models <- readRDS(cfg$models$MIMIC_mmHg$rds)
mdl <- mimic_models[[ cfg$models$MIMIC_mmHg$K ]]
stopifnot(mdl$ng == 4)

pp <- as_tibble(mdl$pprob)
if (!"stay_id" %in% names(pp) && "id" %in% names(pp)) pp <- rename(pp, stay_id = id)
pp_map <- pp %>% transmute(stay_id, Class = normalize_class_levels(class, 4))

# ── 2) PREDICTED PF TRAJECTORIES (safe β extraction) ─────────────────────────
coef_names <- c("intercept","day_period","I(day_period^2)")
get_beta_g <- function(pars, g){
  idx <- vapply(coef_names, function(nm){
    pos <- which(names(pars) == paste0(nm, " class", g))
    if (!length(pos)) stop("coef not found: ", nm, " class", g)
    tail(pos, 1L)  # intercept appears twice; take the latent-process one
  }, integer(1))
  out <- pars[idx]; names(out) <- coef_names; out
}
class_mean_curve_safe <- function(mdl, times){
  p  <- mdl$best
  a0 <- p["Linear 1"]; a1 <- p["Linear 2"]
  X  <- cbind(1, times, times^2); colnames(X) <- coef_names
  map_dfr(seq_len(mdl$ng), function(g){
    b <- get_beta_g(p, g)
    yhat <- as.numeric(a0 + a1 * (X %*% b))
    tibble(Class = normalize_class_levels(g, mdl$ng),
           day_period = times,
           value = yhat,
           type  = "Model predicted")
  })
}
pred_df <- class_mean_curve_safe(mdl, time_seq)

# ── 3) EMPIRICAL PF: MEDIAN + IQR (by class/day) ─────────────────────────────
mimic_pf <- read_csv(cfg$datasets$MIMIC$pf, show_col_types = FALSE) %>%
  filter(between(day_period, 0, admin)) %>%
  inner_join(pp_map, by = "stay_id")

emp_df <- mimic_pf %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(n = n(),
            value = median(pf_ratio_avg, na.rm = TRUE),
            .groups = "drop") %>%
  filter(n >= 10) %>%
  mutate(type = "Empirical median") %>%
  select(Class, day_period, value, type)

iqr_min_n <- 10
emp_summ <- mimic_pf %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(
    n      = n(),
    meanPF = mean(pf_ratio_avg, na.rm = TRUE),
    q25    = quantile(pf_ratio_avg, 0.25, na.rm = TRUE, type = 7),
    q75    = quantile(pf_ratio_avg, 0.75, na.rm = TRUE, type = 7),
    .groups = "drop"
  ) %>%
  filter(n >= iqr_min_n)

# ── 4) PF PLOTS ──────────────────────────────────────────────────────────────
# (A) Empirical median (solid) vs Predicted (dashed)
plot_df <- bind_rows(emp_df, pred_df) %>% arrange(Class, day_period, type)

    p_pf <- ggplot(plot_df, aes(day_period, value,
                            colour = Class, linetype = type,
                            group = interaction(Class, type))) +
  geom_line(linewidth = 1.1, na.rm = TRUE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Empirical median" = "solid",
                                   "Model predicted" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(title = "MIMIC: Empirical (solid) vs Model predicted (dashed) PF by class (K=4)",
       x = "Day", y = "PF Ratio (mmHg)", colour = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_PF_Emp_vs_Pred_K4.png"),
       p_pf, width = 10, height = 6, dpi = 300)

# (B) IQR ribbon + empirical mean (solid) + model predicted (dashed black)
p_pf_iqr <- ggplot() +
  geom_ribbon(
    data = emp_summ,
    aes(x = day_period, ymin = q25, ymax = q75, fill = Class, group = Class),
    alpha = 0.18, show.legend = FALSE
  ) +
  geom_line(
    data = emp_summ,
    aes(day_period, meanPF, color = Class, group = Class),
    linewidth = 1.1
  ) +
  geom_line(
    data = pred_df,
    aes(day_period, value, group = Class),
    linewidth = 1.0, linetype = "dashed", color = "black"
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(
    title = "MIMIC: Empirical mean (solid) with IQR ribbon vs Model predicted (dashed), K=4",
    x = "Day", y = "PF Ratio (mmHg)", color = "Class"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom") +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_PF_Emp_IQR_vs_Pred_K4.png"),
       p_pf_iqr, width = 10, height = 6, dpi = 300)

# (C) NEW: Predicted PF by class only (solid, coloured)
p_pf_pred_only <- ggplot(pred_df,
                         aes(day_period, value, color = Class, group = Class)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(title = "MIMIC: Model‑predicted PF trajectories by class (K=4)",
       x = "Day", y = "PF Ratio (mmHg)", color = "Class") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_PF_PredictedOnly_K4.png"),
       p_pf_pred_only, width = 10, height = 6, dpi = 300)

# ── NEW: Single-panel GAM-smoothed mean PF with model-based 95% CI ───────────
suppressPackageStartupMessages({ library(mgcv) })

# choose your threshold (you set 20 below; change if you prefer 10)
iqr_min_n <- 20

# Aggregate to daily means + weights (subjects per day/class)
agg_mean <- mimic_pf %>%
  dplyr::filter(dplyr::between(day_period, 0, admin), pf_ratio_avg <= 470) %>%
  dplyr::group_by(Class, day_period) %>%
  dplyr::summarise(
    n = dplyr::n_distinct(stay_id),
    meanPF = mean(pf_ratio_avg, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::filter(n >= iqr_min_n) %>%
  dplyr::mutate(Class = factor(Class, levels = class_levels))

# Fit per-class GAM on daily means (weighted), predict ONLY on observed days,
# and add segment ids so lines/ribbons break across gaps.
pred_gam_days <- purrr::map_dfr(levels(agg_mean$Class), function(cl) {
  df <- dplyr::filter(agg_mean, Class == cl) %>% arrange(day_period)
  if (nrow(df) < 10) return(NULL)
  
  # a slightly more flexible smoother
  fit <- mgcv::gam(meanPF ~ s(day_period, k = 6, bs = "cs"),
                   data = df, weights = n, method = "REML")
  
  
  nd <- df %>% select(day_period)  # predict only where data exist (no bridging)
  p  <- predict(fit, nd, se.fit = TRUE)
  
  out <- tibble::tibble(
    Class = factor(cl, levels = class_levels),
    day_period = nd$day_period,
    fit = as.numeric(p$fit),
    lo  = as.numeric(p$fit - 1.96 * p$se.fit),
    hi  = as.numeric(p$fit + 1.96 * p$se.fit)
  ) %>%
    arrange(day_period) %>%
    # break lines/ribbons wherever days are non-consecutive
    mutate(seg = cumsum(dplyr::lag(day_period, default = first(day_period)) != day_period - 1L))
  
  out
})

p_pf_gam_onepanel <- ggplot() +
  geom_ribbon(
    data = pred_gam_days,
    aes(day_period, ymin = lo, ymax = hi, fill = Class, group = interaction(Class, seg)),
    alpha = 0.18, show.legend = FALSE
  ) +
  geom_line(
    data = pred_gam_days,
    aes(day_period, fit, color = Class, group = interaction(Class, seg)),
    linewidth = 1.2
  ) +
  # OPTIONAL: keep/remove model-pred dashed black overlay
  geom_line(data = pred_df, aes(day_period, value, group = Class),
             linewidth = 1.0, linetype = "dashed", color = "black") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(75, 400)) +
  labs(title = "MIMIC: GAM-smoothed mean PF with model-based 95% CI (no extrapolation)",
       x = "Day after AHRF onset", y = "PF Ratio (mmHg)", color = "Class") +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    axis.text  = element_text(size = 18),
    axis.title = element_text(size = 18),
    legend.position = "bottom"
  )

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_PF_GAM_OnePanel_NoExtrap_K4.png"),
       p_pf_gam_onepanel, width = 10, height = 8, dpi = 300)

# optionally print to viewer
p_pf_gam_onepanel


# ── 5) SURVIVAL/DISCHARGE — EMPIRICAL (coloured) vs PREDICTED (black) ────────
# Read stays → tte/event in case they're not already available
parse_datetime <- function(datetime_str) {
  datetime_str <- as.character(datetime_str)
  datetime_str <- trimws(datetime_str)
  datetime_str[datetime_str == ""] <- NA
  suppressWarnings(as.POSIXct(datetime_str, format = "%d/%m/%Y %H:%M", tz = "UTC"))
}
stays_path <- cfg$datasets$MIMIC$stay %||% "Data/mimic_stays.csv"
stopifnot(file.exists(stays_path))
stays <- read_csv(stays_path, show_col_types = FALSE) %>%
  mutate(
    timezero = parse_datetime(timezero),
    outtime  = parse_datetime(outtime)
  ) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(
    raw_tte = as.numeric(difftime(outtime, timezero, units = "days")),
    status  = if_else(icu_mort == 1, 1L, 2L),      # 1=death, 2=discharge
    event   = if_else(raw_tte > admin, 0L, status),
    tte     = pmin(raw_tte, admin)
  ) %>%
  select(stay_id, tte, event)

# Join class → stays
emp_base <- stays %>% inner_join(pp_map, by = "stay_id")

# Predicted CIF/S from model
get_pred_curves <- function(mdl, times){
  ci_all <- as.data.frame(lcmm::cuminc(mdl, time = times, draws = FALSE)[[1]])
  zero   <- data.frame(
    event = rep(1:2, each = 1), time = 0,
    as.data.frame(matrix(0, nrow = 2, ncol = mdl$ng,
                         dimnames = list(NULL, paste0("class", 1:mdl$ng))))
  )
  bind_rows(zero, ci_all) %>%
    mutate(Event = factor(event, levels = 1:2, labels = c("Death","Discharge"))) %>%
    pivot_longer(starts_with("class"), names_to = "ClassVar", values_to = "CIF") %>%
    transmute(
      Class = normalize_class_levels(gsub("^class", "Class ", ClassVar), mdl$ng),
      time  = as.numeric(time),
      Event,
      Y     = ifelse(Event == "Death", 1 - CIF, CIF),   # Survival for death; CIF for discharge
      Source = "Predicted"
    ) %>%
    arrange(Class, Event, time)
}
pred_curves <- get_pred_curves(mdl, time_seq_cif)

# Empirical CIF/S from data
get_emp_curves <- function(base_df){
  split(base_df, base_df$Class) %>%
    purrr::imap_dfr(function(df, clab){
      if (!nrow(df)) return(NULL)
      stopifnot(all(df$event %in% c(0,1,2)))
      ci <- cmprsk::cuminc(df$tte, df$event)   # 1=death, 2=discharge
      death <- if (!is.null(ci$`1 1`))
        tibble(time = ci$`1 1`$time, Y = 1 - ci$`1 1`$est, Event = "Death") else tibble()
      disc  <- if (!is.null(ci$`1 2`))
        tibble(time = ci$`1 2`$time, Y =      ci$`1 2`$est, Event = "Discharge") else tibble()
      bind_rows(
        tibble(time = 0, Y = 1, Event = "Death"),
        tibble(time = 0, Y = 0, Event = "Discharge"),
        death, disc
      ) %>%
        mutate(Class = factor(clab, levels = class_levels),
               Source = "Empirical") %>%
        arrange(Event, time)
    })
}
emp_curves <- get_emp_curves(emp_base)

# (A) Faceted overlay — empirical (coloured) vs predicted (black)
p_surv_cmp <- ggplot() +
  geom_line(
    data = emp_curves %>% filter(Event == "Death"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "solid", na.rm = TRUE
  ) +
  geom_line(
    data = emp_curves %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "dashed", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves %>% filter(Event == "Death"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "solid", color = "black", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "dashed", color = "black", na.rm = TRUE
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(
    title = "MIMIC: Empirical (coloured) vs Predicted (black) — Survival (solid) & Discharge (dashed)",
    x = "Days from start", y = "Survival / Discharge incidence",
    color = "Class"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom") +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_SurvDisc_Pred_vs_Emp_K4.png"),
       p_surv_cmp, width = 10, height = 7, dpi = 300)

# (B) Single‑panel, predicted‑only survival (colour by class)
p_surv_pred_only <- ggplot() +
  geom_line(
    data = pred_curves %>% filter(Event == "Death"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.2, linetype = "solid", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.2, linetype = "dashed", na.rm = TRUE
  ) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "MIMIC: Predicted Survival (solid) & Discharge CIF (dashed) by Class (K=4)",
       x = "Day", y = "Survival / Discharge incidence",
       color = "Class") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_SurvDisc_PredictedOnly_K4.png"),
       p_surv_pred_only, width = 10, height = 7, dpi = 300)

# Survival (solid) = 1 - CIF(death); Discharge (dashed) = CIF(discharge)

emp_surv_disc <- emp_curves %>%
  dplyr::filter(Event %in% c("Death", "Discharge")) %>%
  dplyr::mutate(
    Class = factor(Class, levels = class_levels),
    # keep Y as-is: for Event == "Death" this is SURVIVAL (already 1 - CIF(death))
    # for Discharge it is the CIF(discharge)
    PlotSeries = dplyr::recode(Event,
                               "Death" = "Survival",
                               "Discharge" = "Discharge CIF")
  )

p_surv_disc_onepanel <- ggplot(
  emp_surv_disc,
  aes(x = time, y = Y, color = Class, linetype = PlotSeries,
      group = interaction(Class, PlotSeries))
) +
  geom_step(linewidth = 1.1, na.rm = TRUE, direction = "hv") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Survival" = "solid", "Discharge CIF" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "MIMIC: Observed Survival (solid) & Discharge CIF (dashed) by Class — single panel",
       x = "Day after onset AHRF", y = "Survival / Discharge incidence",
       color = "Class", linetype = NULL) +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    axis.text  = element_text(size = 18),
    axis.title = element_text(size = 18),
    legend.position = "bottom"
  )

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_Survival_and_Discharge_Observed_OnePanel.png"),
       p_surv_disc_onepanel, width = 10, height = 8, dpi = 300)

# Print to viewer
p_surv_disc_onepanel
# =====================================================================
# ICHT: load, classify with MIMIC model, and plot PF & survival overlays
# (append-only; uses the same palette and helpers already defined)
# =====================================================================

suppressPackageStartupMessages({ library(lubridate) })  # for parse_dt_icht

# --- parser for ICHT timestamps (from your pipeline) -----------------
parse_dt_icht <- function(x, tz="UTC"){
  if (inherits(x,"POSIXct")) return(x)
  xc <- trimws(as.character(x)); out <- rep(NA_real_, length(xc))
  tf <- function(idx,f){ if(!any(idx))return();
    out[idx]<<-suppressWarnings(as.numeric(f(xc[idx], tz=tz, quiet=TRUE))) }
  tf(is.na(out), ymd_hms); tf(is.na(out), ymd_hm)
  tf(is.na(out), dmy_hms); tf(is.na(out), dmy_hm)
  num <- is.na(out)&grepl("^\\d+$",xc); if(any(num)) out[num]<-as.numeric(xc[num])
  as.POSIXct(out, origin="1970-01-01", tz)
}

# --- stays: compute tte/event for ICHT --------------------------------
stays_path_icht <- cfg$datasets$ICHT$stay %||% "Data/icht_stays.csv"
stopifnot(file.exists(stays_path_icht))

stays_icht <- readr::read_csv(stays_path_icht, show_col_types = FALSE) %>%
  mutate(
    timezero = parse_dt_icht(timezero),
    outtime  = parse_dt_icht(outtime)
  ) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(
    raw_tte = as.numeric(difftime(outtime, timezero, units = "days")),
    status  = if_else(icu_mort == 1, 1L, 2L),      # 1=death, 2=discharge
    event   = if_else(raw_tte > admin, 0L, status),
    tte     = pmin(raw_tte, admin)
  ) %>%
  select(stay_id, tte, event)

# --- PF daily for ICHT ------------------------------------------------
pf_path_icht <- cfg$datasets$ICHT$pf %||% "Data/icht_pf_daily.csv"
stopifnot(file.exists(pf_path_icht))

pf_icht_raw <- readr::read_csv(pf_path_icht, show_col_types = FALSE) %>%
  filter(dplyr::between(day_period, 0, admin))

# --- joint long data (PF up to each patient's tte) -------------------
joint_icht <- pf_icht_raw %>%
  left_join(stays_icht, by = "stay_id") %>%
  filter(!is.na(tte), day_period <= floor(tte))

# --- assign classes to ICHT using the MIMIC model --------------------
# ensure all model-needed columns exist
needed_vars <- unique(unlist(mdl$Xnames, use.names = FALSE))
missing_vars <- setdiff(needed_vars, names(joint_icht))
for (v in missing_vars) joint_icht[[v]] <- 0

upd_icht <- update(mdl, data = as.data.frame(joint_icht),
                   B = mdl$best, ng = mdl$ng, maxiter = 0, verbose = FALSE)

prob_cols <- grep("^prob", colnames(upd_icht$pprob), value = TRUE)[1:mdl$ng]

class_map_icht <- tibble(stay_id = unique(joint_icht$stay_id)) %>%
  bind_cols(as_tibble(upd_icht$pprob[, prob_cols, drop = FALSE])) %>%
  mutate(Class = normalize_class_levels(max.col(dplyr::select(., all_of(prob_cols))), mdl$ng)) %>%
  select(stay_id, Class)

# --- build empirical PF summaries for ICHT ---------------------------
pf_icht <- pf_icht_raw %>%
  inner_join(class_map_icht, by = "stay_id")

# PF (median per day/class) for solid line
emp_df_icht <- pf_icht %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(n = n(),
            value = median(pf_ratio_avg, na.rm = TRUE),
            .groups = "drop") %>%
  filter(n >= 10) %>%
  mutate(type = "Empirical median") %>%
  select(Class, day_period, value, type)

# PF (mean + IQR) for ribbons
iqr_min_n <- 10
emp_summ_icht <- pf_icht %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(
    n      = n(),
    meanPF = mean(pf_ratio_avg, na.rm = TRUE),
    q25    = quantile(pf_ratio_avg, 0.25, na.rm = TRUE, type = 7),
    q75    = quantile(pf_ratio_avg, 0.75, na.rm = TRUE, type = 7),
    .groups = "drop"
  ) %>%
  filter(n >= iqr_min_n)

# --- PF plots for ICHT ------------------------------------------------
# (1) Empirical median (solid) vs Model predicted (dashed)
plot_df_icht <- bind_rows(emp_df_icht, pred_df) %>% arrange(Class, day_period, type)

p_pf_icht <- ggplot(plot_df_icht, aes(day_period, value,
                                      colour = Class, linetype = type,
                                      group = interaction(Class, type))) +
  geom_line(linewidth = 1.1, na.rm = TRUE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Empirical median" = "solid",
                                   "Model predicted" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(title = "ICHT: Empirical (solid) vs Model predicted (dashed) PF by class (K=4)",
       x = "Day", y = "PF Ratio (mmHg)", colour = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "ICHT_PF_Emp_vs_Pred_K4.png"),
       p_pf_icht, width = 10, height = 6, dpi = 300)

# (2) IQR ribbon + empirical mean (solid) + model predicted (dashed black)
p_pf_iqr_icht <- ggplot() +
  geom_ribbon(
    data = emp_summ_icht,
    aes(x = day_period, ymin = q25, ymax = q75, fill = Class, group = Class),
    alpha = 0.18, show.legend = FALSE
  ) +
  geom_line(
    data = emp_summ_icht,
    aes(day_period, meanPF, color = Class, group = Class),
    linewidth = 1.1
  ) +
  geom_line(
    data = pred_df,
    aes(day_period, value, group = Class),
    linewidth = 1.0, linetype = "dashed", color = "black"
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(
    title = "ICHT: Empirical mean (solid) with IQR ribbon vs Model predicted (dashed), K=4",
    x = "Day", y = "PF Ratio (mmHg)", color = "Class"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA), 
        legend.position = "bottom") +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )

ggsave(file.path(cfg$out_dir %||% ".", "ICHT_PF_Emp_IQR_vs_Pred_K4.png"),
       p_pf_iqr_icht, width = 10, height = 6, dpi = 300)

# --- Survival/Discharge: empirical (ICHT) vs predicted (model) ---------------
# Reuse model-predicted curves (cohort-agnostic)
pred_curves_icht <- get_pred_curves(mdl, time_seq_cif)

# Build empirical CIF/S for ICHT
# Build empirical CIF/S for ICHT (fixed)
emp_curves_icht <- stays_icht %>%
  inner_join(class_map_icht, by = "stay_id") %>%
  # make sure Class is a factor with the standard levels
  mutate(Class = factor(Class, levels = class_levels)) %>%
  # now the dot pronoun refers to this data frame
  split(.$Class) %>%
  purrr::imap_dfr(function(df, clab){
    if (!nrow(df)) return(NULL)
    stopifnot(all(df$event %in% c(0,1,2)))
    ci <- cmprsk::cuminc(df$tte, df$event)  # 1=death, 2=discharge
    
    death <- if (!is.null(ci$`1 1`))
      tibble(time = ci$`1 1`$time, Y = 1 - ci$`1 1`$est, Event = "Death") else tibble()
    disc  <- if (!is.null(ci$`1 2`))
      tibble(time = ci$`1 2`$time, Y =      ci$`1 2`$est, Event = "Discharge") else tibble()
    
    bind_rows(
      tibble(time = 0, Y = 1, Event = "Death"),
      tibble(time = 0, Y = 0, Event = "Discharge"),
      death, disc
    ) %>%
      mutate(Class = factor(clab, levels = class_levels),
             Source = "Empirical") %>%
      arrange(Event, time)
  })

# Faceted overlay for ICHT (same styling as MIMIC)
p_surv_cmp_icht <- ggplot() +
  geom_line(
    data = emp_curves_icht %>% filter(Event == "Death"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "solid", na.rm = TRUE
  ) +
  geom_line(
    data = emp_curves_icht %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "dashed", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves_icht %>% filter(Event == "Death"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "solid", color = "black", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves_icht %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "dashed", color = "black", na.rm = TRUE
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(
    title = "ICHT: Empirical (coloured) vs Predicted (black) — Survival (solid) & Discharge (dashed)",
    x = "Days from start", y = "Survival / Discharge incidence",
    color = "Class"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom") +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )
ggsave(file.path(cfg$out_dir %||% ".", "ICHT_SurvDisc_Pred_vs_Emp_K4.png"),
       p_surv_cmp_icht, width = 10, height = 7, dpi = 300)


# ── NEW: Single-panel smoothed observed PF with 95% CI (MIMIC) ────────────────
# ── One-panel: IQR ribbons + smoothed observed trajectories (MIMIC) ───────────
library(mgcv)

iqr_min_n <- 20

# Aggregate to daily means + weights (subjects per day/class)
agg_mean <- mimic_pf %>%
  dplyr::filter(dplyr::between(day_period, 0, admin), pf_ratio_avg <= 470) %>%
  dplyr::group_by(Class, day_period) %>%
  dplyr::summarise(
    n = dplyr::n_distinct(stay_id),
    meanPF = mean(pf_ratio_avg, na.rm = TRUE),
    .groups = "drop"
  ) %>% dplyr::filter(n >= iqr_min_n) %>%
  dplyr::mutate(Class = factor(Class, levels = class_levels))

# Fit per-class GAM on the daily means, weighted by n
pred_gam <- purrr::map_dfr(levels(agg_mean$Class), function(cl) {
  df <- dplyr::filter(agg_mean, Class == cl)
  if (nrow(df) < 5) return(NULL)
  
  fit <- mgcv::gam(meanPF ~ s(day_period, k = 6, bs = "cs"),
                   data = df, weights = n, method = "REML")
  
  # ⬇️ only predict within observed range for this class
  nd <- tibble::tibble(day_period = seq(min(df$day_period), max(df$day_period), by = 1))
  
  p  <- predict(fit, nd, se.fit = TRUE)
  tibble::tibble(
    Class = factor(cl, levels = class_levels),
    day_period = nd$day_period,
    fit = as.numeric(p$fit),
    lo  = as.numeric(p$fit - 1.96 * p$se.fit),
    hi  = as.numeric(p$fit + 1.96 * p$se.fit)
  )
})
p_onepanel_gam <- ggplot() +
  geom_ribbon(data = pred_gam,
              aes(day_period, ymin = lo, ymax = hi, fill = Class, group = Class),
              alpha = 0.18, show.legend = FALSE) +
  geom_line(data = pred_gam,
            aes(day_period, fit, color = Class, group = Class),
            linewidth = 1.2) +
  # optional: add model-pred dashed black
  geom_line(data = pred_df, aes(day_period, value, group = Class),
            linewidth = 1.0, linetype = "dashed", color = "black") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(75, 400)) +
  labs(title = "MIMIC: GAM-smoothed mean PF with model-based 95% CI",
       x = "Day", y = "PF Ratio (mmHg)", color = "Class") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom") ; p_onepanel_gam


emp_onepanel <- emp_curves %>%
  dplyr::filter(Event %in% c("Death")) %>%
  dplyr::mutate(Class = factor(Class, levels = class_levels))

p_surv_emp_onepanel <- ggplot(
  emp_onepanel,
  aes(x = time, y = Y, color = Class, linetype = Event,
      group = interaction(Class, Event))
) +
  geom_step(linewidth = 1.1, na.rm = TRUE, direction = "hv") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Death" = "solid", "Discharge" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "MIMIC: Observed Survival (solid) & Discharge CIF (dashed) by Class — single panel",
       x = "Day", y = "Survival / Discharge incidence",
       color = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "MIMIC_SurvDisc_Observed_OnePanel_K4.png"),
       p_surv_emp_onepanel, width = 10, height = 7, dpi = 300)






emp_onepanel_icht <- emp_curves_icht %>%
  dplyr::filter(Event %in% c("Death")) %>%
  dplyr::mutate(Class = factor(Class, levels = class_levels))

p_surv_emp_onepanel_icht <- ggplot(
  emp_onepanel_icht,
  aes(x = time, y = Y, color = Class, linetype = Event,
      group = interaction(Class, Event))
) +
  geom_step(linewidth = 1.1, na.rm = TRUE, direction = "hv") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Death" = "solid", "Discharge" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "ICHT: Observed Survival (solid) & Discharge CIF (dashed) by Class — single panel",
       x = "Day", y = "Survival / Discharge incidence",
       color = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# AUMC: load, classify with MIMIC model, and plot PF & survival overlays
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({ library(lubridate) })  # for parse_dt_aumc

# --- parser for AUMC timestamps (DD-MM-YYYY HH:MM:SS) ------------------------
parse_dt_aumc <- function(x, tz="UTC"){
  if (inherits(x,"POSIXct")) return(x)
  xc <- trimws(as.character(x)); out <- rep(NA_real_, length(xc))
  tf <- function(idx,f){ if(!any(idx))return();
    out[idx]<<-suppressWarnings(as.numeric(f(xc[idx], tz=tz, quiet=TRUE))) }
  tf(is.na(out), ymd_hms); tf(is.na(out), ymd_hm)
  tf(is.na(out), dmy_hms); tf(is.na(out), dmy_hm)
  num <- is.na(out)&grepl("^\\d+$",xc); if(any(num)) out[num]<-as.numeric(xc[num])
  as.POSIXct(out, origin="1970-01-01", tz)
}

# --- stays: compute tte/event for AUMC --------------------------------
stays_path_aumc <- cfg$datasets$AUMC$stay %||% "Data/umc_stays.csv"
stopifnot(file.exists(stays_path_aumc))

stays_aumc <- readr::read_csv(stays_path_aumc, show_col_types = FALSE) %>%
  mutate(
    timezero = parse_dt_aumc(timezero),
    outtime  = parse_dt_aumc(outtime)
  ) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(
    raw_tte = as.numeric(difftime(outtime, timezero, units = "days")),
    status  = if_else(icu_mort == 1, 1L, 2L),      # 1=death, 2=discharge
    event   = if_else(raw_tte > admin, 0L, status),
    tte     = pmin(raw_tte, admin)
  ) %>%
  select(stay_id, tte, event)

# --- PF daily for AUMC ------------------------------------------------
pf_path_aumc <- cfg$datasets$AUMC$pf %||% "Data/umc_pf_daily.csv"
stopifnot(file.exists(pf_path_aumc))

pf_aumc_raw <- readr::read_csv(pf_path_aumc, show_col_types = FALSE) %>%
  filter(dplyr::between(day_period, 0, admin))

# --- joint long data (PF up to each patient's tte) -------------------
joint_aumc <- pf_aumc_raw %>%
  left_join(stays_aumc, by = "stay_id") %>%
  filter(!is.na(tte), day_period <= floor(tte))

# --- assign classes to AUMC using the MIMIC model --------------------
# ensure all model-needed columns exist
needed_vars <- unique(unlist(mdl$Xnames, use.names = FALSE))
missing_vars <- setdiff(needed_vars, names(joint_aumc))
for (v in missing_vars) joint_aumc[[v]] <- 0

upd_aumc <- update(mdl, data = as.data.frame(joint_aumc),
                   B = mdl$best, ng = mdl$ng, maxiter = 0, verbose = FALSE)

prob_cols <- grep("^prob", colnames(upd_aumc$pprob), value = TRUE)[1:mdl$ng]

class_map_aumc <- tibble(stay_id = unique(joint_aumc$stay_id)) %>%
  bind_cols(as_tibble(upd_aumc$pprob[, prob_cols, drop = FALSE])) %>%
  mutate(Class = normalize_class_levels(max.col(dplyr::select(., all_of(prob_cols))), mdl$ng)) %>%
  select(stay_id, Class)

# --- build empirical PF summaries for AUMC ---------------------------
pf_aumc <- pf_aumc_raw %>%
  inner_join(class_map_aumc, by = "stay_id")

# PF (median per day/class) for solid line
emp_df_aumc <- pf_aumc %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(n = n(),
            value = median(pf_ratio_avg, na.rm = TRUE),
            .groups = "drop") %>%
  filter(n >= 10) %>%
  mutate(type = "Empirical median") %>%
  select(Class, day_period, value, type)

# PF (mean + IQR) for ribbons
iqr_min_n <- 10
emp_summ_aumc <- pf_aumc %>%
  filter(pf_ratio_avg <= 470) %>%
  group_by(Class, day_period) %>%
  summarise(
    n      = n(),
    meanPF = mean(pf_ratio_avg, na.rm = TRUE),
    q25    = quantile(pf_ratio_avg, 0.25, na.rm = TRUE, type = 7),
    q75    = quantile(pf_ratio_avg, 0.75, na.rm = TRUE, type = 7),
    .groups = "drop"
  ) %>%
  filter(n >= iqr_min_n)

# --- PF plots for AUMC ------------------------------------------------
# (1) Empirical median (solid) vs Model predicted (dashed)
plot_df_aumc <- bind_rows(emp_df_aumc, pred_df) %>% arrange(Class, day_period, type)

p_pf_aumc <- ggplot(plot_df_aumc, aes(day_period, value,
                                      colour = Class, linetype = type,
                                      group = interaction(Class, type))) +
  geom_line(linewidth = 1.1, na.rm = TRUE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Empirical median" = "solid",
                                   "Model predicted" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(title = "AUMC: Empirical (solid) vs Model predicted (dashed) PF by class (K=4)",
       x = "Day", y = "PF Ratio (mmHg)", colour = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "AUMC_PF_Emp_vs_Pred_K4.png"),
       p_pf_aumc, width = 10, height = 6, dpi = 300)

# (2) IQR ribbon + empirical mean (solid) + model predicted (dashed black)
p_pf_iqr_aumc <- ggplot() +
  geom_ribbon(
    data = emp_summ_aumc,
    aes(x = day_period, ymin = q25, ymax = q75, fill = Class, group = Class),
    alpha = 0.18, show.legend = FALSE
  ) +
  geom_line(
    data = emp_summ_aumc,
    aes(day_period, meanPF, color = Class, group = Class),
    linewidth = 1.1
  ) +
  geom_line(
    data = pred_df,
    aes(day_period, value, group = Class),
    linewidth = 1.0, linetype = "dashed", color = "black"
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_fill_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = y_breaks_pf, limits = c(0, 400)) +
  labs(
    title = "AUMC: Empirical mean (solid) with IQR ribbon vs Model predicted (dashed), K=4",
    x = "Day", y = "PF Ratio (mmHg)", color = "Class"
  ) +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA), 
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )

ggsave(file.path(cfg$out_dir %||% ".", "AUMC_PF_Emp_IQR_vs_Pred_K4.png"),
       p_pf_iqr_aumc, width = 10, height = 6, dpi = 300)

# --- Survival/Discharge: empirical (AUMC) vs predicted (model) ---------------
# Reuse model-predicted curves (cohort-agnostic)
pred_curves_aumc <- get_pred_curves(mdl, time_seq_cif)

# Build empirical CIF/S for AUMC
emp_curves_aumc <- stays_aumc %>%
  inner_join(class_map_aumc, by = "stay_id") %>%
  mutate(Class = factor(Class, levels = class_levels)) %>%
  split(.$Class) %>%
  purrr::imap_dfr(function(df, clab){
    if (!nrow(df)) return(NULL)
    stopifnot(all(df$event %in% c(0,1,2)))
    ci <- cmprsk::cuminc(df$tte, df$event)  # 1=death, 2=discharge
    
    death <- if (!is.null(ci$`1 1`))
      tibble(time = ci$`1 1`$time, Y = 1 - ci$`1 1`$est, Event = "Death") else tibble()
    disc  <- if (!is.null(ci$`1 2`))
      tibble(time = ci$`1 2`$time, Y =      ci$`1 2`$est, Event = "Discharge") else tibble()
    
    bind_rows(
      tibble(time = 0, Y = 1, Event = "Death"),
      tibble(time = 0, Y = 0, Event = "Discharge"),
      death, disc
    ) %>%
      mutate(Class = factor(clab, levels = class_levels),
             Source = "Empirical") %>%
      arrange(Event, time)
  })

# Faceted overlay for AUMC (same styling as MIMIC/ICHT)
p_surv_cmp_aumc <- ggplot() +
  geom_line(
    data = emp_curves_aumc %>% filter(Event == "Death"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "solid", na.rm = TRUE
  ) +
  geom_line(
    data = emp_curves_aumc %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.1, linetype = "dashed", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves_aumc %>% filter(Event == "Death"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "solid", color = "black", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves_aumc %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, group = Class),
    linewidth = 1.1, linetype = "dashed", color = "black", na.rm = TRUE
  ) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(
    title = "AUMC: Empirical (coloured) vs Predicted (black) — Survival (solid) & Discharge (dashed)",
    x = "Days from start", y = "Survival / Discharge incidence",
    color = "Class"
  ) +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    panel.background = element_rect(fill = NA),
    legend.position = "bottom",
    plot.title = element_text(size = 16, face = "bold"),       # Title size
    axis.title = element_text(size = 18),                      # Axis label size 18
    axis.text = element_text(size = 12),                       # Axis tick size
    strip.text = element_text(size = 13, face = "bold"),       # Facet labels
    legend.text = element_text(size = 12),                     # Legend text
    legend.title = element_text(size = 13)                     # Legend title
  )
ggsave(file.path(cfg$out_dir %||% ".", "AUMC_SurvDisc_Pred_vs_Emp_K4.png"),
       p_surv_cmp_aumc, width = 10, height = 7, dpi = 300)

# --- Single-panel predicted-only survival (colour by class) for AUMC ---------
p_surv_pred_only_aumc <- ggplot() +
  geom_line(
    data = pred_curves_aumc %>% filter(Event == "Death"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.2, linetype = "solid", na.rm = TRUE
  ) +
  geom_line(
    data = pred_curves_aumc %>% filter(Event == "Discharge"),
    aes(x = time, y = Y, color = Class, group = Class),
    linewidth = 1.2, linetype = "dashed", na.rm = TRUE
  ) +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "AUMC: Predicted Survival (solid) & Discharge CIF (dashed) by Class (K=4)",
       x = "Day", y = "Survival / Discharge incidence",
       color = "Class") +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "AUMC_SurvDisc_PredictedOnly_K4.png"),
       p_surv_pred_only_aumc, width = 10, height = 7, dpi = 300)

# --- Single-panel OBSERVED survival & discharge (colour by class) for AUMC ---
emp_onepanel_aumc <- emp_curves_aumc %>%
  filter(Event %in% c("Death")) %>%
  mutate(Class = factor(Class, levels = class_levels))

p_surv_emp_onepanel_aumc <- ggplot(
  emp_onepanel_aumc,
  aes(x = time, y = Y, color = Class, linetype = Event,
      group = interaction(Class, Event))
) +
  geom_step(linewidth = 1.1, na.rm = TRUE, direction = "hv") +
  scale_color_manual(values = class_cols_named, drop = FALSE) +
  scale_linetype_manual(values = c("Death" = "solid", "Discharge" = "dashed")) +
  scale_x_continuous(breaks = x_breaks, limits = c(0, admin)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  labs(title = "AUMC: Observed Survival (solid) & Discharge CIF (dashed) by Class — single panel",
       x = "Day", y = "Survival / Discharge incidence",
       color = "Class", linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir %||% ".", "AUMC_SurvDisc_Observed_OnePanel_K4.png"),
       p_surv_emp_onepanel_aumc, width = 10, height = 7, dpi = 300)

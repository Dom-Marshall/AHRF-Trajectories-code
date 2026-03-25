# =============================================================================
# Overlay of MODEL-PREDICTED trajectories (PF & optional Survival) for
# MIMIC vs ICHT CR-LCMMs — no empirical data used
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr)
  library(ggplot2); library(lcmm)
})

# --------- CONFIG -------------------------------------------------------------
cfg <- list(
  admin_days = 14,
  out_dir    = "outputs",
  models = list(
    MIMIC_mmHg = list(
      rds   = "cr_models/mimic_multiclass_joint_models_full_14d.rds",
      unit  = "mmHg",
      K     = 4,
      label = "MIMIC"
    ),
    ICHT_kPa = list(
      rds   = "cr_models/icht_multiclass_joint_models_full_14d.rds",
      unit  = "kPa",
      K     = 4,
      label = "ICHT"
    )
  ),
  # keep your chosen mapping to align ICHT classes to MIMIC classes
  class_map_icht_to_mimic = c("1"=1, "2"=4, "3"=2, "4"=3),
  make_survival = TRUE
)

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

# --------- units --------------------------------------------------------------
conv <- 7.50062
to_mmHg <- function(x, from_unit) if (from_unit == "mmHg") x else if (from_unit == "kPa") x*conv else stop("unit?")
safe_readRDS <- function(p){ stopifnot(file.exists(p)); readRDS(p) }

# --------- SAFE helper to get class-level predicted PF curves -----------------
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
  purrr::map_dfr(seq_len(mdl$ng), function(g){
    b <- get_beta_g(p, g)
    yhat <- as.numeric(a0 + a1 * (X %*% b))
    tibble(Class = factor(paste0("Class ", g), levels = paste0("Class ", 1:mdl$ng)),
           day_period = times,
           PF_model_unit = yhat)
  })
}

# --------- Load models --------------------------------------------------------
mdl_M <- safe_readRDS(cfg$models$MIMIC_mmHg$rds)[[ cfg$models$MIMIC_mmHg$K ]]
mdl_I <- safe_readRDS(cfg$models$ICHT_kPa$rds)[[  cfg$models$ICHT_kPa$K   ]]
stopifnot(mdl_M$ng == mdl_I$ng, mdl_M$ng == cfg$models$MIMIC_mmHg$K)

K       <- mdl_M$ng
admin   <- cfg$admin_days
times   <- 0:admin
classes <- paste0("Class ", 1:K)

# Named scales (avoid :=)
col_vals <- setNames(c("deepskyblue","tomato"),
                     c(cfg$models$MIMIC_mmHg$label, cfg$models$ICHT_kPa$label))
lt_vals  <- setNames(c("solid","dashed"),
                     c(cfg$models$MIMIC_mmHg$label, cfg$models$ICHT_kPa$label))

# --------- Predicted PF (common mmHg scale) -----------------------------------
pf_M <- class_mean_curve_safe(mdl_M, times) %>%
  transmute(Model = cfg$models$MIMIC_mmHg$label,
            Class, day_period,
            PF_mmHg = to_mmHg(PF_model_unit, cfg$models$MIMIC_mmHg$unit))

pf_I_raw <- class_mean_curve_safe(mdl_I, times) %>%
  transmute(Model = cfg$models$ICHT_kPa$label,
            Class, day_period,
            PF_mmHg = to_mmHg(PF_model_unit, cfg$models$ICHT_kPa$unit))

# Map ICHT classes to MIMIC alignment
map_vec <- cfg$class_map_icht_to_mimic
pf_I <- pf_I_raw %>%
  mutate(Class_num = as.integer(sub("^Class\\s+", "", Class)),
         Class_mapped_num = as.integer(map_vec[as.character(Class_num)]),
         Class = factor(paste0("Class ", Class_mapped_num), levels = classes)) %>%
  select(-Class_num, -Class_mapped_num)

pf_both <- bind_rows(pf_M, pf_I) %>%
  mutate(Model = factor(Model, levels = names(col_vals)))

p_pf <- ggplot(pf_both, aes(x = day_period, y = PF_mmHg,
                            color = Model, linetype = Model,
                            group = interaction(Model, Class))) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Class, nrow = 2, drop = FALSE) +
  scale_color_manual(values = col_vals) +
  scale_linetype_manual(values = lt_vals) +
  scale_x_continuous(breaks = seq(0, admin, 2), limits = c(0, admin)) +
  scale_y_continuous(breaks = c(50,100,150,200,250,300,350,400), limits = c(0, 400)) +
  labs(title = "Model predicted PF trajectories — MIMIC vs ICHT (aligned classes)",
       x = "Day", y = "PF (mmHg)", color = NULL, linetype = NULL) +
  theme_minimal() +
  theme(panel.border = element_rect(color = "black", fill = NA),
        panel.background = element_rect(fill = NA),
        legend.position = "bottom")

ggsave(file.path(cfg$out_dir, "Pred_PF_MIMIC_vs_ICHT.png"),
       p_pf, width = 10, height = 6, dpi = 300)

# --------- OPTIONAL: Predicted survival/discharge overlays --------------------
if (isTRUE(cfg$make_survival)) {
  get_pred_surv <- function(mdl, times, model_label){
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
        Model = model_label,
        Class = factor(gsub("^class", "Class ", ClassVar), levels = classes),
        time  = as.numeric(time),
        Event,
        Y     = ifelse(Event == "Death", 1 - CIF, CIF)
      ) %>%
      arrange(Class, Event, time)
  }
  
  times_cif <- seq(1, admin, by = 1)
  
  sv_M <- get_pred_surv(mdl_M, times_cif, cfg$models$MIMIC_mmHg$label)
  sv_I_raw <- get_pred_surv(mdl_I, times_cif, cfg$models$ICHT_kPa$label)
  
  sv_I <- sv_I_raw %>%
    mutate(Class_num = as.integer(sub("^Class\\s+", "", Class)),
           Class_mapped_num = as.integer(map_vec[as.character(Class_num)]),
           Class = factor(paste0("Class ", Class_mapped_num), levels = classes)) %>%
    select(-Class_num, -Class_mapped_num)
  
  sv_both <- bind_rows(sv_M, sv_I) %>%
    mutate(Model = factor(Model, levels = names(col_vals)))
  
  p_surv <- ggplot() +
    geom_line(
      data = sv_both %>% filter(Event == "Death"),
      aes(x = time, y = Y, color = Model, group = interaction(Model, Class)),
      linewidth = 1.1, linetype = "solid", na.rm = TRUE
    ) +
    geom_line(
      data = sv_both %>% filter(Event == "Discharge"),
      aes(x = time, y = Y, color = Model, group = interaction(Model, Class)),
      linewidth = 1.1, linetype = "dashed", na.rm = TRUE
    ) +
    facet_wrap(~ Class, nrow = 2, drop = FALSE) +
    scale_color_manual(values = col_vals) +
    scale_x_continuous(breaks = seq(0, admin, 2), limits = c(0, admin)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    labs(title = "Model predicted Survival (solid) & Discharge (dashed) — MIMIC vs ICHT",
         x = "Days from start", y = "Survival / Discharge incidence", color = NULL) +
    theme_minimal() +
    theme(panel.border = element_rect(color = "black", fill = NA),
          panel.background = element_rect(fill = NA),
          legend.position = "bottom")
  
  ggsave(file.path(cfg$out_dir, "Pred_SurvDisc_MIMIC_vs_ICHT.png"),
         p_surv, width = 10, height = 7, dpi = 300)
}

# =============================================================================
# END
# =============================================================================

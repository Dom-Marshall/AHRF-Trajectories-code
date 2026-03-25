# ================================================================
#     Sensitivity: MIMIC vs ARDS-labelled MIMIC   –   v2025-06-23
# ================================================================
# 0 · PACKAGES ----------------------------------------------------
suppressPackageStartupMessages({
  library(lcmm);    library(dplyr);   library(tidyr)
  library(purrr);   library(readr);   library(ggplot2)
  library(lubridate)
})

## 1 · HELPERS ----------------------------------------------------
conv <- 7.50062               # mmHg ↔ kPa
unit_conv <- \(x,from,to) if (from==to) x else if (from=="mmHg"&&to=="kPa") x/conv else
  if (from=="kPa" &&to=="mmHg") x*conv else
    stop("unsupported units")

safe_readRDS <- \(p){
  if (!file.exists(p)) stop("file not found:\n  ", p)
  readRDS(p)
}

class_mean_curve <- function(mdl, times){
  K <- mdl$ng
  coefs <- mdl$best
  a0 <- coefs["Linear 1"]; a1 <- coefs["Linear 2"]
  X  <- cbind(1, times, times^2)
  map_dfr(1:K, \(g){
    b <- coefs[paste0(c("intercept","day_period","I(day_period^2)"), " class", g)]
    tibble(day_period = times,
           class      = g,
           PF_pred_mmHg = a0 + a1*as.numeric(X %*% b))
  })
}

parse_dt_mimic <- function(datetime_str) {
  datetime_str <- as.character(datetime_str)
  datetime_str <- trimws(datetime_str)
  datetime_str[datetime_str == ""] <- NA
  suppressWarnings(as.POSIXct(datetime_str, format = "%d/%m/%Y %H:%M", tz = "UTC"))
}

tidy_pp <- function(pprob_matrix) {
  # Convert posterior probability matrix to tidy tibble
  # Assumes first column is stay_id and remaining are probYT1, probYT2, etc.
  df <- as.data.frame(pprob_matrix)
  
  # Get column names
  prob_cols <- grep("^probYT", names(df), value = TRUE)
  
  # Find class with max probability
  if (length(prob_cols) > 0) {
    df$class <- apply(df[, prob_cols, drop = FALSE], 1, which.max)
  }
  
  as_tibble(df)
}

## 2 · CONFIG  -----------------------------------------------------
admin_days   <- 14
mimic_model  <- list(rds = "cr_models/mimic_multiclass_joint_models_full_14d.rds",
                     unit = "mmHg", K = 4)

paths <- list(
  stays      = "Data/mimic_stays.csv",
  pf_daily   = "Data/mimic_pf_daily.csv",
  ards_lbl   = "Data/ARDS_labels.csv",
  mimic_key  = "Data/mimic_key.csv"
)

out_dir <- "ARDS sensitivity analysis2"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

## 3 · DATA  -------------------------------------------------------
# --- MIMIC stays (baseline + outcome)
stay_all <- read_csv(paths$stays, show_col_types = FALSE) |>
  mutate(across(c(timezero, outtime), parse_dt_mimic)) |>
  filter(!is.na(timezero) & !is.na(outtime) & outtime > timezero) |>
  mutate(raw_tte = as.numeric(difftime(outtime, timezero, units = "days")),
         status  = if_else(icu_mort == 1, 1L, 2L),
         event   = if_else(raw_tte > admin_days, 0L, status),
         tte     = pmin(raw_tte, admin_days)) |>
  select(stay_id, tte, event)

# --- ARDS label → stay_id list
ards_label <- read_csv(paths$ards_lbl, show_col_types = FALSE) |>
  filter(ARDS == 1) |>
  left_join(read_csv(paths$mimic_key, show_col_types = FALSE), by = "hadm_id") |>
  distinct(stay_id)

non_ards_ahrf_label <- read_csv(paths$ards_lbl, show_col_types = FALSE) |>
  filter(Not_ARDS == 1) |>
  left_join(read_csv(paths$mimic_key, show_col_types = FALSE), by = "hadm_id") |>
  distinct(stay_id)

# --- helper: join PF + outcome, restrict to first 14 days -----------
make_joint <- \(stay_tbl){
  read_csv(paths$pf_daily, show_col_types = FALSE) |>
    filter(day_period <= admin_days) |>
    left_join(stay_tbl, by = "stay_id") |>
    filter(!is.na(tte) & day_period <= floor(tte))
}

# ------------ 3·A  BUILD DATASETS  -----------------------------------------
datasets <- list(
  MIMIC_noARDS = make_joint(stay_all |> semi_join(non_ards_ahrf_label, by = "stay_id")),
  MIMIC_ARDS   = make_joint(stay_all |> semi_join(ards_label,        by = "stay_id"))
)

# ------------ 3·B  POSTERIOR PROBABILITIES  --------------------------------
mdl  <- safe_readRDS(mimic_model$rds)[[ mimic_model$K ]]
pp_all <- tidy_pp(mdl$pprob)                           # all MIMIC stays

pp_noards <- pp_all |> semi_join(non_ards_ahrf_label, by = "stay_id")
pp_ards   <- pp_all |> semi_join(ards_label,          by = "stay_id")

pprob <- list(
  MIMIC_noARDS = pp_noards,
  MIMIC_ARDS   = pp_ards
)



## 5 · ANALYSES  ====================================================
# ------------------------------------------------------------------
# Make a "data + class" version of each cohort once
# ------------------------------------------------------------------
data_cls <- setNames(
  lapply(names(datasets), function(cohort){
    datasets[[cohort]] |>
      left_join(pprob[[cohort]] |> select(stay_id, class), by = "stay_id")
  }),
  names(datasets)
)

# A · Median PF trajectories ---------------------------------------
traj_median <- imap_dfr(data_cls, \(df, cohort){
  df |>
    group_by(class, day_period) |>
    summarise(PF_med = median(pf_ratio_avg, na.rm = TRUE),
              .groups = "drop") |>
    mutate(cohort = cohort)
})

p_traj <- ggplot(traj_median,
                 aes(day_period, PF_med,
                     colour = cohort, linetype = cohort)) +
  geom_smooth(linewidth = 1, se=F) +
  facet_wrap(~ class, nrow = 2) +
  scale_color_manual(values = c(MIMIC_noARDS = "steelblue",
                                MIMIC_ARDS = "firebrick")) +
  scale_linetype_manual(values = c(MIMIC_noARDS = "solid",
                                   MIMIC_ARDS = "dashed")) +
  labs(title = "Median PF trajectories (MIMIC overall vs ARDS subset)",
       x = "Day from ICU entry", y = "PF (mmHg)") +
  theme_bw()

ggsave(file.path(out_dir,
                 "PF_median_traj_MIMIC_vs_ARDS.png"),
       p_traj, width = 10, height = 5)

# B · Posterior certainty ------------------------------------------
plot_pp_hist <- function(pp, cohort, K = mimic_model$K){
  # keep only prob1 ... probK
  prob_cols <- paste0("probYT", 1:K)
  
  max_p <- pp |>
    select(all_of(prob_cols)) |>
    as.matrix() |>
    apply(1, max)
  
  ggplot(tibble(p_max = max_p), aes(p_max)) +
    geom_histogram(binwidth = 0.05, boundary = 0, closed = "left",
                   fill = "grey70", colour = "white") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
    labs(title = paste("Posterior certainty –", cohort),
         x = "Max posterior probability", y = "Patients") +
    theme_minimal()
}

imap(pprob, function(pp, cohort){
  p <- plot_pp_hist(pp, cohort)
  ggsave(file.path(out_dir, paste0("PP_hist_", cohort, ".png")),
         p, width = 6, height = 4)
})

pp_descr <- imap_dfr(pprob, function(pp, cohort){
  prob_cols <- paste0("probYT", 1:mimic_model$K)
  
  p_max <- pp |>
    select(all_of(prob_cols)) |>
    as.matrix() |>
    apply(1, max)
  
  tibble(
    cohort        = cohort,
    n             = length(p_max),
    mean_pmax     = mean(p_max),
    median_pmax   = median(p_max),
    Q1            = quantile(p_max,  .25),
    Q3            = quantile(p_max,  .75),
    pct_ge_0.80   = mean(p_max >= .8) * 100
  )
})

write_csv(pp_descr,
          file.path(out_dir, "Posterior_pmax_descriptives.csv"))



# C · 14-day ICU mortality -----------------------------------------
mort_tbl <- imap_dfr(data_cls, function(df, cohort){
  df |>
    distinct(stay_id, class, event) |>
    group_by(class) |>
    summarise(
      cohort    = cohort,
      N_total   = n(),
      N_death   = sum(event == 1),
      Mort14_pct = mean(event == 1) * 100,
      .groups   = "drop"
    )
})

write_csv(mort_tbl,
          file.path(out_dir, "Mortality_MIMIC_vs_ARDS.csv"))

# D · Empirical vs model-predicted PF ------------------------------
pred_curve <- class_mean_curve(mdl, 0:admin_days)

rmse_tbl <- imap_dfr(data_cls, \(df, cohort){
  df |>
    left_join(pred_curve, by = c("class", "day_period")) |>
    group_by(class) |>
    summarise(cohort,
              RMSE = sqrt(mean((pf_ratio_avg - PF_pred_mmHg)^2,
                               na.rm = TRUE)),
              .groups = "drop")
})

write_csv(rmse_tbl,
          file.path(out_dir, "RMSE_byClass_MIMIC_vs_ARDS.csv"))

# E · Class-distribution comparison ---------------------------------
class_cnts <- imap_dfr(pprob, \(pp, cohort){
  pp |>
    count(class, name = "N") |>
    mutate(cohort,
           Prop = N / sum(N))
})

write_csv(class_cnts,
          file.path(out_dir, "Class_distribution_MIMIC_vs_ARDS.csv"))

p_cls <- ggplot(class_cnts,
                aes(factor(class), Prop,
                    fill = cohort)) +
  geom_col(position = "dodge", width = .7) +
  scale_fill_manual(values = c(MIMIC_noARDS = "steelblue",
                               MIMIC_ARDS = "firebrick")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Class distribution\nMIMIC overall vs ARDS subset",
       x = "Class", y = "Percentage of patients",
       fill = NULL) +
  theme_minimal()

ggsave(file.path(out_dir,
                 "Class_distribution_MIMIC_vs_ARDS.png"),
       p_cls, width = 6, height = 4)

# --------------------------------------------------------------
#  E2 · ARDS vs Non-ARDS composition within each class
# --------------------------------------------------------------
# Create a combined dataset with ARDS status for each patient
ards_composition <- bind_rows(
  pprob$MIMIC_ARDS |> mutate(ARDS_status = "ARDS"),
  pprob$MIMIC_noARDS |> mutate(ARDS_status = "Non-ARDS")
) |>
  count(class, ARDS_status) |>
  group_by(class) |>
  mutate(total = sum(n),
         proportion = n / total,
         pct_label = paste0(round(proportion * 100, 1), "%")) |>
  ungroup()

# Save the data
write_csv(ards_composition,
          file.path(out_dir, "ARDS_composition_by_class.csv"))

# Create an attractive stacked bar chart
p_ards_comp <- ggplot(ards_composition,
                      aes(x = factor(class), y = proportion, fill = ARDS_status)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.8) +
  geom_text(aes(label = paste0(n, "\n(", pct_label, ")")),
            position = position_stack(vjust = 0.5),
            colour = "white", fontface = "bold", size = 4) +
  scale_fill_manual(values = c("ARDS" = "#d73027", 
                                "Non-ARDS" = "#4575b4"),
                    name = "AHRF Category") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = c(0, 0)) +
  labs(title = "ARDS vs Non-ARDS Composition by Class",
       subtitle = "Numbers and percentages shown for each group",
       x = "Class", 
       y = "Proportion of Patients") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, colour = "gray40"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave(file.path(out_dir, "ARDS_composition_by_class.png"),
       p_ards_comp, width = 8, height = 6, dpi = 300)

# Alternative: Grouped bar chart (side-by-side comparison)
p_ards_grouped <- ggplot(ards_composition,
                         aes(x = factor(class), y = proportion, fill = ARDS_status)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, 
           colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(n, "\n", pct_label)),
            position = position_dodge(width = 0.8),
            vjust = -0.3, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("ARDS" = "#d73027", 
                                "Non-ARDS" = "#4575b4"),
                    name = "AHRF Category") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title = "ARDS vs Non-ARDS Distribution Across Classes",
       subtitle = "Side-by-side comparison",
       x = "Class", 
       y = "Proportion within Class") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, colour = "gray40"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave(file.path(out_dir, "ARDS_composition_by_class_grouped.png"),
       p_ards_grouped, width = 8, height = 6, dpi = 300)

# --------------------------------------------------------------
#  E3 · Alternative view: Class breakdown WITHIN each ARDS status
#       (i.e., "Of all ARDS patients, what % are in each class?")
# --------------------------------------------------------------
# Calculate proportions within each ARDS status (separate denominators)
ards_class_breakdown <- bind_rows(
  pprob$MIMIC_ARDS |> mutate(ARDS_status = "ARDS"),
  pprob$MIMIC_noARDS |> mutate(ARDS_status = "Non-ARDS")
) |>
  count(ARDS_status, class) |>
  group_by(ARDS_status) |>
  mutate(total_in_group = sum(n),
         proportion = n / total_in_group,
         pct_label = paste0(round(proportion * 100, 1), "%")) |>
  ungroup()

# Save the data
write_csv(ards_class_breakdown,
          file.path(out_dir, "Class_breakdown_within_ARDS_status.csv"))

# Create side-by-side bar chart comparing class distributions
p_class_breakdown <- ggplot(ards_class_breakdown,
                            aes(x = factor(class), y = proportion, fill = ARDS_status)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7,
           colour = "white", linewidth = 0.5) +
  geom_text(aes(label = pct_label),
            position = position_dodge(width = 0.8),
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("ARDS" = "#d73027", 
                                "Non-ARDS" = "#4575b4"),
                    name = "AHRF Category") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Class Distribution by AHRF Category",
       subtitle = "Proportions calculated separately for ARDS and Non-ARDS patients",
       x = "Class", 
       y = "Proportion within AHRF Category") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, colour = "gray40"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave(file.path(out_dir, "Class_breakdown_within_ARDS_status.png"),
       p_class_breakdown, width = 8, height = 6, dpi = 300)

# Alternative: Faceted view showing each AHRF category separately
p_class_breakdown_facet <- ggplot(ards_class_breakdown,
                                  aes(x = factor(class), y = proportion, fill = ARDS_status)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(n, "\n(", pct_label, ")")),
            vjust = -0.3, size = 3.5, fontface = "bold", colour = "gray20") +
  facet_wrap(~ ARDS_status, ncol = 2) +
  scale_fill_manual(values = c("ARDS" = "#d73027", 
                                "Non-ARDS" = "#4575b4")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Class Distribution Within Each AHRF Category",
       subtitle = "Each panel sums to 100%",
       x = "Class", 
       y = "Proportion of Patients") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, colour = "gray40"),
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 12),
    strip.background = element_rect(fill = "gray95", colour = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave(file.path(out_dir, "Class_breakdown_within_ARDS_status_faceted.png"),
       p_class_breakdown_facet, width = 9, height = 5, dpi = 300)

# Create a summary table showing the numbers clearly
summary_table <- ards_class_breakdown |>
  select(ARDS_status, class, n, proportion) |>
  mutate(percentage = paste0(round(proportion * 100, 1), "%")) |>
  pivot_wider(id_cols = class,
              names_from = ARDS_status,
              values_from = c(n, percentage),
              names_glue = "{ARDS_status}_{.value}") |>
  select(class, 
         `ARDS_n`, `ARDS_percentage`,
         `Non-ARDS_n`, `Non-ARDS_percentage`)

write_csv(summary_table,
          file.path(out_dir, "Class_breakdown_comparison_table.csv"))

# --------------------------------------------------------------
#  F · ARDS subset: empirical median vs model-predicted PF
# --------------------------------------------------------------
# 1.  Empirical median PF (ARDS only)
med_ards <- data_cls$MIMIC_ARDS |>
  group_by(class, day_period) |>
  summarise(emp_median = median(pf_ratio_avg, na.rm = TRUE),
            .groups = "drop")

# 2.  Model-predicted curve (already int class in pred_curve)
#     → make 'class' the same type on both sides  ★
med_ards   <- med_ards   |> mutate(class = as.integer(as.character(class)))  # ★
pred_curve <- pred_curve |> mutate(class = as.integer(class))                # ★

plot_df <- med_ards |>
  left_join(pred_curve, by = c("class", "day_period")) |>
  rename(pred_PF = PF_pred_mmHg)

# 3.  Plot
p_emp_vs_pred_ards <- ggplot(plot_df, aes(day_period)) +
  geom_line(aes(y = emp_median), colour = "firebrick", linewidth = 1.1) +
  geom_line(aes(y = pred_PF),    colour = "black",    linetype = "dashed") +
  facet_wrap(~ class, nrow = 2) +
  labs(title = "ARDS subset: empirical median PF vs model-predicted trajectory",
       x = "Day (0–14)", y = "PF (mmHg)") +
  theme_bw()

# 4.  Save
ggsave(file.path(out_dir, "Emp_vs_Pred_MIMIC_ARDS.png"),
       p_emp_vs_pred_ards, width = 8, height = 4)

#
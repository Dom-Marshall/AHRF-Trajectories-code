###############################################################################
# AI CLASSIFIER FOR to predict hyperinflammatory phenotype and summarize by trajectory class
###############################################################################
# Credit to Dr Joris Pensier et al for sharing the mddell - https://pubmed.ncbi.nlm.nih.gov/40839098/
# All code below modified from their manuscript/github

library(xgboost)
library(dplyr)
library(data.table)
library(tidyr)

tryCatch(
  setwd("C:/Users/dm4312/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories/First submission code/Hyperinflammatory AI classifier"),
  error = function(e) setwd("D:/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories/First submission code/Hyperinflammatory AI classifier")
)

###############################################################################
# A. LOAD THE PRE-TRAINED MODEL
###############################################################################

# AI_Clarity.model converted from legacy binary to JSON format (xgboost 3.0 → 3.1 compatibility)
model_path <- "AI_Clarity.json"
AI_clarity <- xgb.load(model_path)

# Feature names used during training
feature_names <- c(
  "Bilirubin", "Sodium", "Creatinine", "Albumin", "Hematocrit",
  "WBC", "Glucose", "Platelet", "PaCO2", "PaFi",
  "Temperature", "RespRate", "HeartRate", "SBP",
  "Urine", "MinVent", "Bicarbonate",
  "Vasopressors0",           # original binary indicator
  "Vasopressors"             # dummy column created during one-hot encoding
)

###############################################################################
# B. FUNCTION TO PROCESS A COHORT
###############################################################################

process_cohort <- function(cohort_name, data_file, pprob_file) {
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("Processing:", cohort_name, "\n")
  cat(rep("=", 70), "\n\n", sep = "")
  
  # --------------------------------------------------------------------------- #
  # 1. Load data
  # --------------------------------------------------------------------------- #
  raw_cohort <- read.csv(data_file)
  pprob <- read.csv(pprob_file) %>% select(stay_id, class)
  
  # Filter to day 0 or day 1 (use whichever exists)
  if (0 %in% unique(raw_cohort$day)) {
    cohort <- raw_cohort %>% filter(day == 0)
    day_label <- "day 0"
  } else if (1 %in% unique(raw_cohort$day)) {
    cohort <- raw_cohort %>% filter(day == 1)
    day_label <- "day 1"
  } else {
    stop("Neither day 0 nor day 1 found in the data!")
  }
  
  cat("Loaded", nrow(cohort), "patients at", day_label, "\n")
  
  # --------------------------------------------------------------------------- #
  # 2. Rename columns if needed (different cohorts use different names)
  # --------------------------------------------------------------------------- #
  if (cohort_name == "MIMIC") {
    rename_map <- c(
      WBC      = "White_blood_cell_count",
      PaFi     = "maxPaO2_FiO2_ratio",
      RespRate = "Respiratory_rate",
      HeartRate = "Heart_rate",
      SBP      = "Systolic_blood_pressure",
      Urine    = "Urine_output",
      MinVent  = "Minute_Ventilation"
    )
    cohort <- cohort %>% rename(!!!rename_map)
  } else if (cohort_name == "AUMC") {
    # AUMC uses admissionid instead of stay_id and max_PaFi instead of PaFi
    rename_map <- c(
      stay_id  = "admissionid",
      PaFi     = "max_PaFi",
      RespRate = "Respiratory_rate"
    )
    cohort <- cohort %>% rename(!!!rename_map)
  }
  
  # --------------------------------------------------------------------------- #
  # 3. Impute missing values
  # --------------------------------------------------------------------------- #
  id_cols <- c("stay_id", "day")
  num_cols <- names(cohort)[sapply(cohort, is.numeric) & !(names(cohort) %in% id_cols)]
  
  # Compute column means
  col_means <- cohort %>%
    summarise(across(all_of(num_cols), ~ mean(.x, na.rm = TRUE))) %>%
    as.list()
  
  # Handle Vasopressors separately (keep as 0/1)
  vp_fill <- if (all(is.na(cohort$Vasopressors))) {
    0L
  } else {
    as.integer(round(mean(cohort$Vasopressors, na.rm = TRUE)))
  }
  
  # Apply imputation
  cohort <- cohort %>%
    mutate(
      across(all_of(setdiff(num_cols, "Vasopressors")),
             ~ ifelse(is.na(.x), col_means[[cur_column()]], .x)),
      Vasopressors = ifelse(is.na(Vasopressors), vp_fill, Vasopressors)
    )
  
  cat("Imputed missing values\n")
  
  # --------------------------------------------------------------------------- #
  # 4. Prepare features in model order
  # --------------------------------------------------------------------------- #
  feature_order <- c(
    "Bilirubin", "Sodium", "Creatinine", "Albumin", "Hematocrit",
    "WBC", "Glucose", "Platelet", "PaCO2", "PaFi",
    "Temperature", "RespRate", "HeartRate", "SBP",
    "Urine", "MinVent", "Bicarbonate", "Vasopressors"
  )
  
  # One-hot encode Vasopressors
  cohort$Vasopressors[is.na(cohort$Vasopressors)] <- 0
  cohort <- cohort %>% 
    mutate(Vasopressors = factor(Vasopressors, levels = c(0, 1)))
  
  vaso_mat <- model.matrix(~ Vasopressors - 1, data = cohort)
  cohort <- bind_cols(cohort, as.data.frame(vaso_mat))
  
  # Create prediction dataset
  cohort_pred <- cohort[, feature_names]
  cohort_pred$Vasopressors0 <- as.factor(cohort_pred$Vasopressors0)
  
  cohort_pred <- cohort_pred %>% 
    mutate(across(where(is.character), factor)) %>%
    mutate(across(where(is.factor), ~ as.numeric(.x) - 1))
  
  # Assemble design matrix
  X_mat <- as.matrix(cohort_pred)
  X_mat <- X_mat[, feature_names]
  
  # --------------------------------------------------------------------------- #
  # 5. Predict hyperinflammatory phenotype
  # --------------------------------------------------------------------------- #
  prob_vec <- predict(AI_clarity, X_mat)
  
  setDT(cohort_pred)
  cohort_pred[, `:=`(
    stay_id = cohort$stay_id,
    prob_hyper = prob_vec,
    phenotype = factor(ifelse(prob_vec < 0.5, 1, 2), levels = c(1, 2))
  )]
  
  cat("Generated predictions\n")
  
  # --------------------------------------------------------------------------- #
  # 6. Join with trajectory class
  # --------------------------------------------------------------------------- #
  cohort_pred <- cohort_pred %>% 
    left_join(pprob, by = 'stay_id')
  
  cat("Joined with trajectory classes\n")
  
  # --------------------------------------------------------------------------- #
  # 7. Calculate summaries
  # --------------------------------------------------------------------------- #
  
  # Overall summary
  overall_summ <- cohort_pred %>%
    summarise(
      cohort = cohort_name,
      n_total = n(),
      n_hypo = sum(phenotype == 1),
      n_hyper = sum(phenotype == 2),
      pct_hypo = 100 * n_hypo / n_total,
      pct_hyper = 100 * n_hyper / n_total
    )
  
  # By class summary
  class_summ <- cohort_pred %>%
    group_by(class) %>%
    summarise(
      cohort = cohort_name,
      n_total = n(),
      n_hypo = sum(phenotype == 1),
      n_hyper = sum(phenotype == 2),
      pct_hypo = 100 * n_hypo / n_total,
      pct_hyper = 100 * n_hyper / n_total,
      .groups = 'drop'
    ) %>%
    arrange(class)
  
  # --------------------------------------------------------------------------- #
  # 8. Print results
  # --------------------------------------------------------------------------- #
  
  cat("\n--- OVERALL SUMMARY ---\n")
  cat(sprintf("Total N: %d\n", overall_summ$n_total))
  cat(sprintf("Hypoinflammatory (1): %d (%.1f%%)\n", 
              overall_summ$n_hypo, overall_summ$pct_hypo))
  cat(sprintf("Hyperinflammatory (2): %d (%.1f%%)\n", 
              overall_summ$n_hyper, overall_summ$pct_hyper))
  
  cat("\n--- BY TRAJECTORY CLASS ---\n")
  for (i in 1:nrow(class_summ)) {
    row <- class_summ[i, ]
    cat(sprintf("\nClass %d (n=%d):\n", row$class, row$n_total))
    cat(sprintf("  Hypoinflammatory:  %d (%.1f%%)\n", row$n_hypo, row$pct_hypo))
    cat(sprintf("  Hyperinflammatory: %d (%.1f%%)\n", row$n_hyper, row$pct_hyper))
  }
  
  # Return results
  return(list(
    cohort_name = cohort_name,
    data = cohort_pred,
    overall = overall_summ,
    by_class = class_summ
  ))
}

###############################################################################
# C. PROCESS ALL THREE COHORTS
###############################################################################

mimic_results <- process_cohort(
  cohort_name = "MIMIC",
  data_file = "mimic_ahrf_features_v31.csv",
  pprob_file = "pprob_MIMIC.csv"
)

icht_results <- process_cohort(
  cohort_name = "ICHT",
  data_file = "ICHT_daily_averages_by_stay.csv",
  pprob_file = "pprob_ICHT.csv"
)

aumc_results <- process_cohort(
  cohort_name = "AUMC",
  data_file = "aumc_features_ai_classifier.csv",
  pprob_file = "pprob_AUMC.csv"
)

###############################################################################
# D. CREATE COMBINED SUMMARY TABLES
###############################################################################

cat("\n", rep("=", 70), "\n", sep = "")
cat("COMBINED SUMMARY TABLES\n")
cat(rep("=", 70), "\n\n", sep = "")

# Overall comparison
overall_combined <- bind_rows(
  mimic_results$overall,
  icht_results$overall,
  aumc_results$overall
)

cat("--- OVERALL: MIMIC vs ICHT vs AUMC ---\n")
print(overall_combined %>% 
        select(cohort, n_total, n_hypo, pct_hypo, n_hyper, pct_hyper), 
      row.names = FALSE)

# By class comparison
class_combined <- bind_rows(
  mimic_results$by_class,
  icht_results$by_class,
  aumc_results$by_class
)

cat("\n--- BY CLASS: MIMIC vs ICHT vs AUMC ---\n")
print(class_combined %>% 
        select(cohort, class, n_total, n_hypo, pct_hypo, n_hyper, pct_hyper), 
      row.names = FALSE)

###############################################################################
# E. CREATE FORMATTED TABLES FOR MANUSCRIPT
###############################################################################

# Table 1: Overall summary with n (%)
overall_table <- overall_combined %>%
  mutate(
    Hypoinflammatory = sprintf("%d (%.1f%%)", n_hypo, pct_hypo),
    Hyperinflammatory = sprintf("%d (%.1f%%)", n_hyper, pct_hyper)
  ) %>%
  select(Cohort = cohort, N = n_total, Hypoinflammatory, Hyperinflammatory)

cat("\n--- TABLE 1: Overall Phenotype Distribution ---\n")
print(overall_table, row.names = FALSE)

# Table 2: By class with n (%)
class_table <- class_combined %>%
  mutate(
    Hypoinflammatory = sprintf("%d (%.1f%%)", n_hypo, pct_hypo),
    Hyperinflammatory = sprintf("%d (%.1f%%)", n_hyper, pct_hyper)
  ) %>%
  select(Cohort = cohort, Class = class, N = n_total, 
         Hypoinflammatory, Hyperinflammatory)

cat("\n--- TABLE 2: Phenotype Distribution by Trajectory Class ---\n")
print(class_table, row.names = FALSE)

###############################################################################
# F. SAVE RESULTS
###############################################################################

# Save detailed data
write.csv(mimic_results$data, "MIMIC_predictions_with_class.csv", row.names = FALSE)
write.csv(icht_results$data, "ICHT_predictions_with_class.csv", row.names = FALSE)
write.csv(aumc_results$data, "AUMC_predictions_with_class.csv", row.names = FALSE)

# Save summary tables
write.csv(overall_combined, "Overall_phenotype_summary.csv", row.names = FALSE)
write.csv(class_combined, "By_class_phenotype_summary.csv", row.names = FALSE)
write.csv(overall_table, "Table_overall_formatted.csv", row.names = FALSE)
write.csv(class_table, "Table_by_class_formatted.csv", row.names = FALSE)

cat("\n", rep("=", 70), "\n", sep = "")
cat("RESULTS SAVED\n")
cat(rep("=", 70), "\n\n", sep = "")
cat("Files saved:\n")
cat("  - MIMIC_predictions_with_class.csv\n")
cat("  - ICHT_predictions_with_class.csv\n")
cat("  - AUMC_predictions_with_class.csv\n")
cat("  - Overall_phenotype_summary.csv\n")
cat("  - By_class_phenotype_summary.csv\n")
cat("  - Table_overall_formatted.csv\n")
cat("  - Table_by_class_formatted.csv\n")

cat("\nAnalysis complete!\n")

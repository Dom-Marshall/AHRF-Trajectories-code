# Load necessary libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tableone)
  library(flextable)
})

# Set working directories as needed
 setwd("C:/Users/dm4312/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories")

# ───── Load and preprocess MIMIC data ─────
df_mimic <- read.csv("Data/processed_mimic_table1.csv")  # Placeholder for already-processed df
df_mimic$cohort <- "MIMIC-IV"

# ───── Load and preprocess ICHT data ─────
df_icht <- read.csv("Data/processed_icht_table1.csv")  # Placeholder
df_icht$cohort <- "ICHT"
df_icht$vaso_running[df_icht$norad_equiv_d1 > 0] <- 1
df_icht$vaso_running[is.na(df_icht$norad_equiv_d1)] <- 0

# Placeholder: if comorbidities are not yet in ICHT, create dummy columns for alignment
comorb_vars <- c("myocardial_infarct", "congestive_heart_failure", "cerebrovascular_disease", 
                 "chronic_pulmonary_disease", "renal_disease", "com_diabetes", 
                 "com_cancer", "com_liver")

for (v in comorb_vars) {
  if (!(v %in% colnames(df_icht))) df_icht[[v]] <- NA
}

# ───── Harmonize variable names ─────
df_mimic <- df_mimic %>%
  mutate(age = admission_age,
         gender = gender, 
         icu_mortality = icu_mort,
         icu_los = los_icu,
         norepinephrine_equivalent = norepinephrine_equivalent_ifon,
         pf_ratio = avg_pf_0_24,
         peep = avg_peep_0_24,
         paco2 = avg_pco2,
         ph = avg_ph,
         lactate = avg_lactate,
         bicarbonate = avg_bicarbonate,
         creatinine = creatinine,
         wbc = wbc,
         heart_rate = avg_heart_rate,
         mbp = com_avg_mbp,
         resp_rate = avg_resp_rate,
         temperature = avg_temperature,
         vaso_running = vaso_running)

df_icht <- df_icht %>%
  mutate(age = age,
         gender = sex ,
         icu_mortality = icu_mort,
         norepinephrine_equivalent = norad_equiv_d1,
         pf_ratio = pf_ratio_mmHg_d1,
         heart_rate = avg_hr_d1 ,
         temperature = temp_d1,
         peep = peep_d1   ,
         paco2 = pco2_mmHg_d1,
         resp_rate = rr_mean_d1,
         ph = pH_d1,
         lactate = lactate_d1,
         bicarbonate = bicarb_d1,
         creatinine = creatinine_mgdl_d1,
         wbc = wbc_d1,
         mbp = mean_bp_d1 
         )

# Combine datasets
common_vars <- intersect(names(df_mimic), names(df_icht))
df_combined <- bind_rows(df_mimic[, common_vars], df_icht[, common_vars])
df_combined$cohort <- factor(df_combined$cohort, levels = c("MIMIC-IV", "ICHT"))

# Convert binary variables to factors for tableone
binaryVars <- c("gender", "icu_mortality", "vaso_running", comorb_vars)
df_combined[binaryVars] <- lapply(df_combined[binaryVars], factor)

# Specify continuous variables
continuousVars <- c("age", "icu_los", "norepinephrine_equivalent", "pf_ratio", "peep", "paco2", 
                    "ph", "lactate", "bicarbonate", "creatinine", "wbc", "heart_rate", "mbp", 
                    "resp_rate", "temperature")

# Generate Table 1
table1 <- CreateTableOne(vars = c(binaryVars, continuousVars), strata = "cohort", data = df_combined)
table_df <- print(table1, nonnormal = TRUE, quote = FALSE, noSpaces = TRUE) %>%
  as.data.frame() %>%
  tibble::rownames_to_column(var = "Variable")

# Save and display
write.csv(table_df, "combined_table1.csv", row.names = FALSE)

flextable(table_df) %>%
  theme_vanilla() %>%
  set_caption("Descriptive Table 1: MIMIC-IV vs ICHT") %>%
  autofit()


colnames(df_icht)

df_icht$sex <- as.factor(df_icht$gender)
df_icht$icu_mort <- as.factor(df_icht$icu_mort)
df_icht$vaso_running <- as.factor(df_icht$vaso_running)

binaryVars2 <- c("sex", "icu_mort", "vaso_running")
continuousVars2 <-c("age", "icu_los",
  "avg_hr_d1",
  "mean_bp_d1",
  "rr_mean_d1",
  "temp_d1",
  "norad_equiv_d1",
  "pf_ratio_mmHg_d1",          "pf_ratio_mmHg_d2" ,         "pf_ratio_mmHg_d3",
  "peep_d1",                   "peep_d2" ,                  "peep_d3",
  "pH_d1",                     "pco2_mmHg_d1" , 
  "lactate_d1" ,               "bicarb_d1"   ,
  "creatinine_mgdl_d1", "platelets_d1"    , "wbc_d1",
  "sofa_cardio", "sofa_renal", "sofa_total" )
  

table2 <- CreateTableOne(vars = c(binaryVars2, continuousVars2), strata = "class", data = df_icht)
table2_df <- print(table2, nonnormal = TRUE, quote = FALSE, noSpaces = TRUE) %>%
  as.data.frame() %>%
  tibble::rownames_to_column(var = "Variable")
write.csv(table2_df, "icht_supp.csv", row.names = FALSE)

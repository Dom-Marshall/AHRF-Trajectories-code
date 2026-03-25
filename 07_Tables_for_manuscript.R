# ===========================
# AHRF Trajectories – Tables
# End-to-end build script
# ===========================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(stringr); library(purrr)
  library(crosstable); library(flextable)
  library(lubridate); library(tibble); library(arrow)
})

tryCatch(
  setwd("C:/Users/dm4312/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories"),
  error = function(e) setwd("D:/Dropbox/PhD/Manuscripts/AHRF PF Trajectory/AHRF-Trajectories")
)

`%||%` <- function(a,b) if (!is.null(a)) a else b

# =============
# 0) File paths
# =============
paths <- list(
  # class labels
  prob_mim   = "Data/pprob_MIMIC.csv",
  prob_icht  = "Data/pprob_ICHT.csv",
  prob_aumc  = "Data/pprob_AUMC.csv",
  # long data
  long_mim   = "Data/mimic_dynamic_var.csv",
  add_mim    = "Data/mimic_static_var.csv",
  long_icht  = "Data/icht_dynamic_var.csv",
  add_icht   = "Data/icht_static_var.csv",
  aumc_daily = "Data/aumc_daily.csv",
  aumc_pf    = "Data/umc_pf_daily.csv",
  aumc_demo  = "Data/aumc_demo.csv",
  # SOFA
  sofa_mim   = "Data/mimic_3dc_sofa.csv",
  # stays/outcomes
  stays_mim  = "Data/mimic_stays.csv",
  stays_icht = "Data/icht_stays.csv",
  stays_aumc = "Data/umc_stays.csv"
)

class_levels <- paste0("C", 1:4)

# ============================
# 1) Read + harmonise long data
# ============================
# ----- MIMIC
prob_mim <- read_csv(paths$prob_mim, show_col_types = FALSE) %>% select(stay_id, class)
long_mim <- read_csv(paths$long_mim, show_col_types = FALSE) %>%
  mutate(age = as.numeric(admission_age),
         gender_bin = if_else(gender == "M", 1, 0)) %>%
  left_join(prob_mim, by = "stay_id") %>%
  mutate(class = factor(paste0("C", class), levels = class_levels))

# ----- ICHT
prob_icht <- read_csv(paths$prob_icht, show_col_types = FALSE) %>% select(stay_id, class)
long_icht <- read_csv(paths$long_icht, show_col_types = FALSE) %>%
  select(-c(avg_pco2, avg_pf_ratio)) %>%  # drop conflicting cols if present
  mutate(age = as.numeric(age),
         gender_bin = if_else(gender %in% c("M","Male",1), 1, 0)) %>%
  rename(
    days_from_start        = day_from_start,
    norad_vasorate         = avg_norad_equiv,           # already µg/kg/min
    avg_lactate            = avg_lactate,
    locf_bicarbonate       = avg_bicarb,
    avg_peak_insp_pressure = avg_pip2,
    avg_peep               = avg_peep,
    avg_pao2fio2ratio      = avg_pf_ratio_mmHg,
    locf_creatinine        = avg_creatinine,
    avg_pco2               = avg_pco2_mmHg,
    avg_minute_volume      = avg_minute_vent,
    avg_resp_rate          = rr
  ) %>%
  mutate(avg_pao2fio2ratio = ifelse(avg_pao2fio2ratio > 470, NA, avg_pao2fio2ratio)) %>%
  left_join(prob_icht, by = "stay_id") %>%
  mutate(class = factor(paste0("C", class), levels = class_levels))

# ----- AUMC
prob_aumc_raw <- read_csv(paths$prob_aumc, show_col_types = FALSE)
prob_aumc <- if ("stay_id" %in% names(prob_aumc_raw)) {
  prob_aumc_raw %>% select(stay_id, class)
} else if ("admissionid" %in% names(prob_aumc_raw)) {
  prob_aumc_raw %>% rename(stay_id = admissionid) %>% select(stay_id, class)
} else stop("pprob_AUMC.csv must have 'stay_id' or 'admissionid'.")

long_aumc  <- read_csv(paths$aumc_daily, show_col_types = FALSE)
pf_aumc    <- read_csv(paths$aumc_pf,    show_col_types = FALSE)
demo_aumc  <- read_csv(paths$aumc_demo,  show_col_types = FALSE)

# normalise names for joins
if (!"admissionid" %in% names(long_aumc) && "stay_id" %in% names(long_aumc)) long_aumc <- rename(long_aumc, admissionid = stay_id)
if (!"day" %in% names(long_aumc) && "day_period" %in% names(long_aumc)) long_aumc <- rename(long_aumc, day = day_period)
if (!"stay_id" %in% names(pf_aumc) && "admissionid" %in% names(pf_aumc)) pf_aumc <- rename(pf_aumc, stay_id = admissionid)
if (!"day_period" %in% names(pf_aumc) && "day" %in% names(pf_aumc))       pf_aumc <- rename(pf_aumc, day_period = day)

# merge PF onto daily
long_aumc <- long_aumc %>% left_join(pf_aumc, by = c("admissionid" = "stay_id", "day" = "day_period"))

# build weight_used: prefer 'weight_impute' if present, else 'weight'; if missing/<=0 → median of available weights
if (!"weight_impute" %in% names(demo_aumc) && "weight" %in% names(demo_aumc)) {
  demo_aumc <- demo_aumc %>% mutate(weight_impute = weight)
}
# compute cohort median from whichever column we have now
med_weight <- median(demo_aumc$weight_impute, na.rm = TRUE)
if (!is.finite(med_weight)) med_weight <- 82.4  # ultimate fallback, shouldn't trigger if any weight present

demo_aumc <- demo_aumc %>%
  mutate(weight_used = ifelse(is.na(weight_impute) | weight_impute <= 0, med_weight, weight_impute))

# attach demo + compute AUMC NE correctly: (norad_vasorate OR norepinephrine)/weight_used  → µg/kg/min
# also bring MAP, WBC, platelets, pH, etc.
long_aumc <- long_aumc %>%
  left_join(demo_aumc %>% transmute(stay_id, sex, age_imputed, weight_used),
            by = c("admissionid" = "stay_id")) %>%
  mutate(
    age        = as.numeric(age_imputed),
    gender_bin = if_else(sex %in% c(1,"M","Male"), 1, 0, missing = NA_real_),
    # norepinephrine is per-minute dose at AUMC (µg/min) → divide by kg
    NE_equiv_mcgkgmin = norepinephrine / weight_used
  ) %>%
  # make WBC robust to column naming differences
  mutate(
    WBC_10e9L = wbc_10e9_l
  ) %>%
  rename(
    stay_id                = admissionid,
    days_from_start        = day,
    avg_lactate            = lactate,
    locf_bicarbonate       = bicarb_mmol_l,
    avg_peak_insp_pressure = ppeak_cmh2o,
    avg_peep               = peep_cmh2o,
    avg_pao2fio2ratio      = pf_ratio_avg,
    locf_creatinine        = creatinine_mg_dl,
    avg_pco2               = pco2_mmhg,
    avg_minute_volume      = minute_vent,
    avg_resp_rate          = resp_rate_bpm,
    MAP_mmHg               = map_mmhg,
    HR_bpm                 = hr_bpm,
    Temp_C                 = temp_c,
    Platelet_10e9L         = platelets_10e9_l,
    pH                     = ph
  ) %>%
  mutate(avg_pao2fio2ratio = ifelse(avg_pao2fio2ratio > 470, NA, avg_pao2fio2ratio)) %>%
  left_join(prob_aumc, by = "stay_id") %>%
  mutate(class = factor(paste0("C", class), levels = class_levels))

# ===========
# 1b) Cleaning
# ===========
fix_nonneg <- function(df, vars){
  for (v in intersect(vars, names(df))) df[[v]] <- ifelse(df[[v]] < 0, NA_real_, df[[v]])
  df
}
nn_vars <- c("avg_lactate","avg_minute_volume","avg_peep","avg_peak_insp_pressure",
             "avg_resp_rate","locf_bicarbonate","locf_creatinine")
long_mim  <- fix_nonneg(long_mim,  nn_vars)
long_icht <- fix_nonneg(long_icht, nn_vars)
long_aumc <- fix_nonneg(long_aumc, nn_vars)

convert_minute_vent <- function(df){
  if (!"avg_minute_volume" %in% names(df)) return(df)
  df$avg_minute_volume <- ifelse(df$avg_minute_volume > 200, df$avg_minute_volume/1000, df$avg_minute_volume)
  df
}
long_mim  <- convert_minute_vent(long_mim)
long_icht <- convert_minute_vent(long_icht)
long_aumc <- convert_minute_vent(long_aumc)

# ICHT creatinine to mg/dL if needed
if ("locf_creatinine" %in% names(long_icht)) long_icht <- long_icht %>% mutate(locf_creatinine = locf_creatinine/88.4)

clip_between <- function(x, lo, hi) ifelse(is.na(x), NA_real_, pmax(pmin(x, hi), lo))
for (nm in c("avg_lactate","avg_pco2","avg_peak_insp_pressure","avg_peep",
             "avg_resp_rate","locf_bicarbonate","avg_pao2fio2ratio")) {
  for (DF in c("long_mim","long_icht","long_aumc")) {
    df <- get(DF); if (!nm %in% names(df)) next
    df[[nm]] <- switch(nm,
      avg_lactate            = clip_between(df[[nm]], 0, 20),
      avg_pco2               = clip_between(df[[nm]], 10, 120),
      avg_peak_insp_pressure = clip_between(df[[nm]], 0, 80),
      avg_peep               = clip_between(df[[nm]], 0, 30),
      avg_resp_rate          = clip_between(df[[nm]], 0, 80),
      locf_bicarbonate       = clip_between(df[[nm]], 5, 50),
      avg_pao2fio2ratio      = clip_between(df[[nm]], 30, 470),
      df[[nm]]
    )
    assign(DF, df)
  }
}

# ================================
# 2) Stays + Day14 outcomes
# ================================
parse_dt_mimic <- function(x){
  x <- trimws(as.character(x)); x[x==""] <- NA
  suppressWarnings(as.POSIXct(x, format="%d/%m/%Y %H:%M", tz="UTC"))
}
parse_dt_generic <- function(x, tz="UTC"){
  x <- as.character(x)
  a <- suppressWarnings(ymd_hms(x, tz=tz, quiet=TRUE))
  b <- suppressWarnings(ymd_hm (x, tz=tz, quiet=TRUE))
  ifelse(!is.na(a), a, b)
}
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

stays_mim <- read_csv(paths$stays_mim, show_col_types = FALSE) %>%
  mutate(timezero = parse_dt_mimic(timezero),
         outtime  = parse_dt_mimic(outtime)) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(raw_tte = as.numeric(difftime(outtime, timezero, units="days"))) %>%
  select(stay_id, raw_tte, icu_mort)

stays_icht <- read_csv(paths$stays_icht, show_col_types = FALSE) %>%
  mutate(timezero = parse_dt_icht(timezero, "UTC"),
         outtime  = parse_dt_icht(outtime,  "UTC")) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(raw_tte = as.numeric(difftime(outtime, timezero, units="days"))) %>%
  select(stay_id, raw_tte, icu_mort)

# For ICHT, override with correct LOS from processed file (already calculated as post-AHRF from SQL)
icht_processed_los <- read_csv(paths$add_icht, show_col_types = FALSE) %>%
  select(stay_id, icu_los_correct = icu_los)
stays_icht <- stays_icht %>%
  left_join(icht_processed_los, by = "stay_id") %>%
  mutate(raw_tte = coalesce(icu_los_correct, raw_tte)) %>%
  select(stay_id, raw_tte, icu_mort)

stays_aumc <- read_csv(paths$stays_aumc, show_col_types = FALSE) %>%
  mutate(timezero = parse_dt_generic(timezero, "UTC"),
         outtime  = parse_dt_generic(outtime,  "UTC")) %>%
  filter(!is.na(timezero), !is.na(outtime), outtime > timezero) %>%
  mutate(raw_tte = as.numeric(difftime(outtime, timezero, units="days"))) %>%
  select(stay_id, raw_tte, icu_mort)

stays_mim  <- stays_mim  %>% mutate(icu_los_days = raw_tte)
stays_icht <- stays_icht %>% mutate(icu_los_days = raw_tte)
stays_aumc <- stays_aumc %>% mutate(icu_los_days = raw_tte)

mk_event_tte <- function(stays, admin=14){
  stays %>%
    mutate(
      status = if_else(icu_mort == 1, 1L, 2L),      # 1=death, 2=discharge
      event  = if_else(raw_tte > admin, 0L, status),
      tte    = pmin(raw_tte, admin)
    )
}
st7_mim   <- mk_event_tte(stays_mim,  admin=7)
st7_icht  <- mk_event_tte(stays_icht, admin=7)
st7_aumc  <- mk_event_tte(stays_aumc, admin=7)

st14_mim  <- mk_event_tte(stays_mim,  admin=14)
st14_icht <- mk_event_tte(stays_icht, admin=14)
st14_aumc <- mk_event_tte(stays_aumc, admin=14)

mk_day_cols <- function(st, day){
  st %>% transmute(
    stay_id,
    !!paste0("Mortality_Day", day)     := as.numeric(event==1 & tte<=day),
    !!paste0("ICU_Discharge_Day", day) := as.numeric(event==2 & tte<=day)
  )
}

d7_mim   <- mk_day_cols(st7_mim,   7)
d7_icht  <- mk_day_cols(st7_icht,  7)
d7_aumc  <- mk_day_cols(st7_aumc,  7)

d14_mim  <- mk_day_cols(st14_mim,  14)
d14_icht <- mk_day_cols(st14_icht, 14)
d14_aumc <- mk_day_cols(st14_aumc, 14)

# =================================
# 3) Day 0 baselines per site
# =================================
baseline_from_long <- function(long_df, site_label, site){
  d0 <- long_df %>% filter(days_from_start == 0)
  # NE source per site:
  NE_col <- if (site == "AUMC") {
    d0$NE_equiv_mcgkgmin   # computed above
  } else {
    d0$norad_vasorate      # MIMIC/ICHT: already µg/kg/min
  }
  # Helper function to safely extract column
  safe_col <- function(df, colname) {
    if (colname %in% names(df)) df[[colname]] else NA_real_
  }
  tibble(
    stay_id            = d0$stay_id,
    Age_years          = d0$age %||% NA_real_,
    Male_sex           = d0$gender_bin %||% NA_real_,
    RR_bpm             = d0$avg_resp_rate %||% NA_real_,
    NE_equiv_mcgkgmin  = NE_col %||% NA_real_,       # site-specific mapping
    PF_mmHg            = d0$avg_pao2fio2ratio %||% NA_real_,
    PaCO2_mmHg         = d0$avg_pco2 %||% NA_real_,
    PEEP_cmH2O         = d0$avg_peep %||% NA_real_,
    PIP_cmH2O          = d0$avg_peak_insp_pressure %||% NA_real_,
    pH                 = safe_col(d0, "pH"),         # available in AUMC
    Lactate_mmolL      = d0$avg_lactate %||% NA_real_,
    Bicarbonate_mmolL  = d0$locf_bicarbonate %||% NA_real_,
    Creatinine_mgdl    = d0$locf_creatinine %||% NA_real_,
    MinuteVent_Lmin    = d0$avg_minute_volume %||% NA_real_,
    # extras for AUMC (HR, MAP, Temp) extracted from renamed columns
    HR_bpm             = safe_col(d0, "HR_bpm"),
    MAP_mmHg           = safe_col(d0, "MAP_mmHg"),
    Temp_C             = safe_col(d0, "Temp_C"),
    WBC_10e9L          = safe_col(d0, "WBC_10e9L"),
    Platelet_10e9L     = safe_col(d0, "Platelet_10e9L"),
    class              = d0$class %||% NA
  ) %>%
  mutate(
    site = site_label,
    DrivingPressure_cmH2O = ifelse(!is.na(PIP_cmH2O) & !is.na(PEEP_cmH2O), PIP_cmH2O - PEEP_cmH2O, NA_real_)
  )
}

base_mim  <- baseline_from_long(long_mim,  "MIMIC", "MIMIC")
base_icht <- baseline_from_long(long_icht, "ICHT",  "ICHT")
base_aumc <- baseline_from_long(long_aumc, "AUMC",  "AUMC")

# For AUMC, add Mean_BP_mmHg as a copy of MAP_mmHg for consistency with table variable names
base_aumc <- base_aumc %>% mutate(Mean_BP_mmHg = MAP_mmHg)

# Attach outcomes (ICU mortality + LOS + D7 + D14)
attach_outcomes <- function(base_df, stays_df, day7_df, day14_df){
  base_df %>%
    left_join(stays_df %>% select(stay_id, icu_mort, icu_los_days), by="stay_id") %>%
    left_join(day7_df, by="stay_id") %>%
    left_join(day14_df, by="stay_id") %>%
    mutate(ICU_mortality = icu_mort %||% NA_real_,
           ICU_LOS_days  = icu_los_days %||% NA_real_)
}
base_mim  <- attach_outcomes(base_mim,  stays_mim,  d7_mim,  d14_mim)
base_icht <- attach_outcomes(base_icht, stays_icht, d7_icht, d14_icht)
base_aumc <- attach_outcomes(base_aumc, stays_aumc, d7_aumc, d14_aumc)

# ── Mechanical Ventilation at Day 0 ──
# MIMIC: Use modality_at_timezero == "IMV" OR "IMV_NEAR"
# IMV = invasive mechanical ventilation at timezero
# IMV_NEAR = near IMV (within threshold of timezero)
if (file.exists("Data/3dc_mimic_vent_att0.csv")) {
  vent_mim <- read_csv("Data/3dc_mimic_vent_att0.csv", show_col_types = FALSE) %>%
    mutate(imv_or_near_imv_at_t0 = as.integer(
      modality_at_timezero %in% c("IMV", "IMV_NEAR")
    )) %>%
    transmute(stay_id,
              Mechanical_ventilation = imv_or_near_imv_at_t0)
  base_mim <- base_mim %>% left_join(vent_mim, by="stay_id")
}

# AUMC: Use mv_invasive from daily data (already computed in long_aumc processing)
# Extract from day 0 of aumc_daily
aumc_vent <- read_csv(paths$aumc_daily, show_col_types = FALSE) %>%
  filter(day == 0) %>%
  transmute(stay_id = admissionid,
            Mechanical_ventilation = as.integer(mv_invasive == 1))
base_aumc <- base_aumc %>% left_join(aumc_vent, by="stay_id")

# ICHT: Use support_cat from vent_daily_labels parquet file
if (file.exists("Data/icht_vent_daily_labels.parquet")) {
  suppressPackageStartupMessages(library(arrow))
  vent_icht <- read_parquet("Data/icht_vent_daily_labels.parquet") %>%
    filter(rel_day == 0) %>%
    transmute(stay_id,
              Mechanical_ventilation = as.integer(support_cat == "Invasive"))
  base_icht <- base_icht %>% left_join(vent_icht, by="stay_id")
} else {
  base_icht <- base_icht %>% mutate(Mechanical_ventilation = NA_integer_)
}

# ── Small-table vitals/labs for MIMIC & ICHT only ──
add_mim  <- read_csv(paths$add_mim,  show_col_types = FALSE)
add_icht <- read_csv(paths$add_icht, show_col_types = FALSE)

mim_small <- add_mim %>% transmute(
  stay_id,
  HR_bpm         = avg_heart_rate,
  Mean_BP_mmHg   = dplyr::coalesce(avg_mbp, avg_mbp_ni),
  Temp_C         = avg_temperature,
  WBC_10e9L      = wbc,
  Platelet_10e9L = platelet
)
icht_small <- add_icht %>% transmute(
  stay_id,
  HR_bpm         = avg_hr_d1,
  Mean_BP_mmHg   = mean_bp_d1,
  Temp_C         = temp_d1,
  WBC_10e9L      = wbc_d1,
  Platelet_10e9L = platelets_d1
)

# For MIMIC & ICHT, drop the baseline NA values for vitals/labs and use the small table data
# Only drop columns that exist in baseline
base_mim  <- base_mim  %>% 
  select(-any_of(c("HR_bpm", "Mean_BP_mmHg", "Temp_C", "WBC_10e9L", "Platelet_10e9L"))) %>%
  left_join(mim_small, by="stay_id")

base_icht <- base_icht %>% 
  select(-any_of(c("HR_bpm", "Mean_BP_mmHg", "Temp_C", "WBC_10e9L", "Platelet_10e9L"))) %>%
  left_join(icht_small, by="stay_id")
# (AUMC intentionally excluded from small table per request)

# ── SOFA & White ethnicity & NE users-only ──
# MIMIC SOFA
sofa_mim <- read_csv(paths$sofa_mim, show_col_types = FALSE)
if ("sofa_24hours" %in% names(sofa_mim)) {
  sofa_mim <- sofa_mim %>% select(stay_id, SOFA_total = sofa_24hours)
} else {
  comps <- c("respiration_24hours","coagulation_24hours","liver_24hours",
             "cardiovascular_24hours","cns_24hours","renal_24hours")
  have  <- comps[comps %in% names(sofa_mim)]
  sofa_mim <- sofa_mim %>%
    mutate(SOFA_total = rowSums(across(all_of(have)), na.rm = TRUE)) %>%
    select(stay_id, SOFA_total)
}
base_mim <- base_mim %>% left_join(sofa_mim, by = "stay_id")

# ICHT SOFA from processed file if present
if ("sofa_total" %in% names(add_icht)) {
  base_icht <- base_icht %>% left_join(add_icht %>% select(stay_id, SOFA_total = sofa_total), by = "stay_id")
}

# MIMIC White ethnicity
if ("ethnicity" %in% names(add_mim)) {
  base_mim <- base_mim %>%
    left_join(add_mim %>% transmute(stay_id, White_ethnicity = as.numeric(ethnicity == "WHITE")), by = "stay_id")
}

# NE columns: users-only dose + vasopressor use binary + pH, platelets, WBC
# For table display, show NE only for vasopressor users (median will be among users only)
# MIMIC: norepinephrine_equivalent (µg/kg/min), pH, platelets, WBC from additional file
if ("norepinephrine_equivalent" %in% names(add_mim)) {
  base_mim <- base_mim %>%
    left_join(add_mim %>% select(stay_id, norepinephrine_equivalent, avg_ph, platelet, wbc), by="stay_id") %>%
    mutate(NE_equiv_mcgkgmin_users = ifelse(!is.na(norepinephrine_equivalent) & norepinephrine_equivalent > 0, norepinephrine_equivalent, NA_real_),
           Vasopressor_use = ifelse(!is.na(norepinephrine_equivalent) & norepinephrine_equivalent > 0, 1, 0),
           pH = avg_ph,
           Platelet_10e9L = platelet,
           WBC_10e9L = wbc) %>%
    select(-norepinephrine_equivalent, -avg_ph, -platelet, -wbc)
}
# ICHT: day1 norad_equiv, pH, platelets, WBC from processed file
# Note: ICHT NA means not on vasopressors (treat NA as 0)
if ("norad_equiv_d1" %in% names(add_icht)) {
  base_icht <- base_icht %>%
    left_join(add_icht %>% select(stay_id, norad_equiv_d1, pH_d1, platelets_d1, wbc_d1), by="stay_id") %>%
    mutate(NE_equiv_mcgkgmin_users = ifelse(!is.na(norad_equiv_d1) & norad_equiv_d1 > 0, norad_equiv_d1, NA_real_),
           Vasopressor_use = ifelse(!is.na(norad_equiv_d1) & norad_equiv_d1 > 0, 1, 0),
           pH = pH_d1,
           Platelet_10e9L = platelets_d1,
           WBC_10e9L = wbc_d1) %>%
    select(-norad_equiv_d1, -pH_d1, -platelets_d1, -wbc_d1)
}
# AUMC: NE from computed NE_equiv_mcgkgmin (pH and Platelet_10e9L already in baseline from daily data)
base_aumc <- base_aumc %>%
  mutate(NE_equiv_mcgkgmin_users = ifelse(!is.na(NE_equiv_mcgkgmin) & NE_equiv_mcgkgmin > 0, NE_equiv_mcgkgmin, NA_real_),
         Vasopressor_use = ifelse(!is.na(NE_equiv_mcgkgmin) & NE_equiv_mcgkgmin > 0, 1, 0))

# ==========================
# 4) Formatting (median [IQR])
# ==========================
dp <- list(
  n_digits=0, pct_digits=1, cont_digits=1,
  Age_years=0, HR_bpm=0, Mean_BP_mmHg=0, RR_bpm=0,
  Temp_C=1, NE_equiv_mcgkgmin=2, NE_equiv_mcgkgmin_users=2,
  PF_mmHg=0, PaCO2_mmHg=0, PEEP_cmH2O=0, pH=2,
  Lactate_mmolL=1, Bicarbonate_mmolL=1, Creatinine_mgdl=2,
  WBC_10e9L=1, Platelet_10e9L=0, ICU_LOS_days=0, MinuteVent_Lmin=1, DrivingPressure_cmH2O=0
)
fmt_num <- function(x, name, default=dp$cont_digits){
  digs <- dp[[name]] %||% default
  if (all(is.na(x))) return("NA")
  formatC(x, format="f", digits=digs, big.mark=",")
}
fmt_pct <- function(p) paste0(formatC(100*p, format="f", digits=dp$pct_digits), "%")
fmt_n_pct1 <- function(x){
  # CRITICAL: n is TOTAL length (cohort size), not just non-missing
  # Missing values are assumed to be 0 (not present)
  n <- length(x)
  n1 <- sum(x==1, na.rm=TRUE)
  pct <- ifelse(n>0, n1/n, NA_real_)
  paste0(n1, " (", ifelse(is.na(pct),"NA",fmt_pct(pct)), ")")
}
fmt_med_iqr <- function(x, name){
  if (!is.numeric(x)) return(NA_character_)
  x <- x[is.finite(x)]
  if (length(x)==0) return("NA")
  q <- quantile(x, probs = c(.25,.5,.75), na.rm = TRUE, names = FALSE)
  paste0(fmt_num(q[2], name), " [", fmt_num(q[1], name), "–", fmt_num(q[3], name), "]")
}
summ_one_var <- function(x, label){
  if (is.null(x) || length(x) == 0) return(NA_character_)
  if (all(x %in% c(0,1,NA))) return(fmt_n_pct1(x))
  xc <- suppressWarnings(as.numeric(x))
  if (all(xc %in% c(0,1,NA))) return(fmt_n_pct1(xc))
  fmt_med_iqr(x, label)
}
summ_many <- function(df, vars, label_map){
  tibble(
    Variable = (label_map[vars] %||% vars),
    n        = sapply(vars, function(v) sum(!is.na(df[[v]]))),
    Value    = sapply(vars, function(v) summ_one_var(df[[v]], v))
  )
}

label_map1 <- c(
  "Age_years"="Age, years","Male_sex"="Male sex, n (%)","ICU_mortality"="ICU mortality, n (%)",
  "ICU_LOS_days"="ICU length of stay, days",
  "HR_bpm"="Heart rate, bpm","Mean_BP_mmHg"="Mean BP, mmHg","Temp_C"="Temperature, °C","RR_bpm"="Respiratory rate, bpm",
  "Mechanical_ventilation"="Invasive mechanical ventilation, n (%)",
  "Vasopressor_use"="Vasopressor use, n (%)",
  "NE_equiv_mcgkgmin_users"="Norepinephrine equiv, µg/kg/min (users)",
  "PF_mmHg"="PaO2/FiO2, mmHg","PaCO2_mmHg"="PaCO2, mmHg",
  "PEEP_cmH2O"="PEEP, cm H2O","pH"="pH","Lactate_mmolL"="Lactate, mmol/L",
  "Bicarbonate_mmolL"="Bicarbonate, mmol/L","Creatinine_mgdl"="Creatinine, mg/dL",
  "WBC_10e9L"="WBC, ×10^9/L","Platelet_10e9L"="Platelet count, ×10^9/L",
  "MinuteVent_Lmin"="Minute Ventilation, L/min","DrivingPressure_cmH2O"="Driving Pressure, cmH2O"
)
label_map2 <- c(
  "Age_years"="Age, years","Male_sex"="Male sex, n (%)","White_ethnicity"="White Ethnicity, n (%)",
  "Hospital_mortality"="Hospital mortality, n (%)","ICU_mortality"="ICU mortality, n (%)","ICU_LOS_days"="ICU length of stay, days",
  "Mortality_Day7"="Mortality – Day 7, n (%)","ICU_Discharge_Day7"="ICU Discharge – Day 7, n (%)",
  "Mortality_Day14"="Mortality – Day 14, n (%)","ICU_Discharge_Day14"="ICU Discharge – Day 14, n (%)",
  "Mechanical_ventilation"="Invasive mechanical ventilation, n (%)",
  "RR_bpm"="Respiratory rate, breaths/min",
  "NE_equiv_mcgkgmin_users"="Norepinephrine equiv, µg/kg/min (users only)",
  "PF_mmHg"="PaO2/FiO2 0–24h, mmHg","PEEP_cmH2O"="PEEP 0–24h, cmH2O","DrivingPressure_cmH2O"="Driving Pressure, cmH2O",
  "pH"="pH","PaCO2_mmHg"="PaCO2, mmHg","MinuteVent_Lmin"="Minute Ventilation, L/min",
  "Lactate_mmolL"="Lactate, mmol/L","Bicarbonate_mmolL"="Bicarbonate, mmol/L","Creatinine_mgdl"="Creatinine, mg/dL",
  "Platelet_10e9L"="Platelet count, ×10^9/L","WBC_10e9L"="WBC, ×10^9/L","SOFA_total"="Total SOFA",
  "Charlson_index"="Charlson comorbidity index","Hyperinflammatory"="Hyperinflammatory, n (%)"
)

# ==============================
# 5) Table 1 - grouped by category
# ==============================
# Group variables into sensible categories
vars_table1 <- list(
  Demographics = c("Age_years", "Male_sex"),
  Outcomes = c("ICU_mortality", "ICU_LOS_days"),
  `Vital Signs` = c("HR_bpm", "Mean_BP_mmHg", "Temp_C", "RR_bpm"),
  `Organ Support` = c("Mechanical_ventilation", "Vasopressor_use", "NE_equiv_mcgkgmin_users"),
  `Respiratory Parameters` = c("PF_mmHg", "PaCO2_mmHg", "PEEP_cmH2O", "MinuteVent_Lmin", "DrivingPressure_cmH2O"),
  `Laboratory Values` = c("pH", "Lactate_mmolL", "Bicarbonate_mmolL", "Creatinine_mgdl", "WBC_10e9L", "Platelet_10e9L")
)

# Flatten into single vector for processing
vars_table1_flat <- unlist(vars_table1, use.names = FALSE)

fmt_table <- function(vars, label_map, site_cols){
  out <- tibble(Variable = (label_map[vars] %||% vars)) %>%
    left_join(summ_many(site_cols[[1]], vars, label_map) %>% select(Variable, Value) %>% rename(!!site_cols[[2]] := Value), by="Variable") %>%
    left_join(summ_many(site_cols[[3]], vars, label_map) %>% select(Variable, Value) %>% rename(!!site_cols[[4]] := Value), by="Variable") %>%
    left_join(summ_many(site_cols[[5]], vars, label_map) %>% select(Variable, Value) %>% rename(!!site_cols[[6]] := Value), by="Variable")
  n_row <- tibble(
    Variable = "n",
    !!site_cols[[2]] := as.character(nrow(site_cols[[1]])),
    !!site_cols[[4]] := as.character(nrow(site_cols[[3]])),
    !!site_cols[[6]] := as.character(nrow(site_cols[[5]]))
  )
  bind_rows(n_row, out)
}

tbl1_df <- fmt_table(vars_table1_flat, label_map1,
                     list(base_mim, "MIMIC", base_icht, "ICHT", base_aumc, "AUMC"))

# Add category headers
tbl1_with_categories <- tibble(Variable = character(), MIMIC = character(), ICHT = character(), AUMC = character())
for (cat_name in names(vars_table1)) {
  # Add category header row
  cat_row <- tibble(Variable = cat_name, MIMIC = "", ICHT = "", AUMC = "")
  tbl1_with_categories <- bind_rows(tbl1_with_categories, cat_row)
  # Add variables in this category
  cat_vars <- vars_table1[[cat_name]]
  cat_data <- tbl1_df %>% filter(Variable %in% label_map1[cat_vars])
  tbl1_with_categories <- bind_rows(tbl1_with_categories, cat_data)
}
# Prepend n row
n_row <- tbl1_df %>% filter(Variable == "n")
tbl1_final <- bind_rows(n_row, tbl1_with_categories)

ft1 <- flextable(tbl1_final) %>%
  bold(i = ~ Variable %in% c("n", names(vars_table1)), j = 1) %>%  # Bold category headers and n
  fontsize(size = 9, part = "all") %>%  # Smaller font to fit page width
  autofit() %>%
  set_caption("Table 1. Baseline (Day 0) characteristics across cohorts (median [IQR]; no statistical testing).")
save_as_docx("Table 1" = ft1, path = file.path(getwd(), "table1_baseline_day0.docx"))

# =======================================================
# 6) Small table (MIMIC & ICHT only): HR / MAP / Temp / WBC / Platelets
# =======================================================
small_vars <- c("HR_bpm","Mean_BP_mmHg","Temp_C","WBC_10e9L","Platelet_10e9L")
label_small <- c("HR_bpm"="Heart rate, bpm","Mean_BP_mmHg"="Mean BP, mmHg","Temp_C"="Temperature, °C",
                 "WBC_10e9L"="WBC, ×10^9/L","Platelet_10e9L"="Platelet count, ×10^9/L")

small_tbl <- tibble(Variable = (label_small[small_vars] %||% small_vars)) %>%
  left_join(summ_many(base_mim,  small_vars, label_small) %>% select(Variable, Value) %>% rename(MIMIC = Value), by="Variable") %>%
  left_join(summ_many(base_icht, small_vars, label_small) %>% select(Variable, Value) %>% rename(ICHT  = Value), by="Variable")

ft_small <- flextable(small_tbl) %>% autofit() %>%
  set_caption("Supplementary: Selected vitals/labs at Day 0 (MIMIC & ICHT; median [IQR]).")
save_as_docx("Supplement – MIMIC & ICHT vitals/labs" = ft_small,
             path = file.path(getwd(), "supp_vitals_mimic_icht.docx"))

# ==============================================
# 7) Class tables (first row = n % of cohort)
# ==============================================
build_by_class_table <- function(df, vars, label_map){
  cls <- levels(df$class)
  
  n_tot <- nrow(df)
  n_per <- sapply(cls, function(cl) sum(df$class==cl, na.rm=TRUE))
  pct_per <- if (n_tot>0) n_per/n_tot else rep(NA_real_, length(n_per))
  
  # Add n row (just the counts)
  n_row <- tibble(Variable="n") %>%
    bind_cols(as_tibble_row(setNames(as.character(n_per), cls)))
  
  # Add n (%) of cohort row
  pct_row <- tibble(Variable="n (%) of cohort") %>%
    bind_cols(as_tibble_row(setNames(paste0(n_per, " (", fmt_pct(pct_per), ")"), cls)))
  
  # Build rows one at a time
  all_rows <- list(n_row, pct_row)
  for (v in vars){
    # Check if variable exists
    if (!v %in% names(df)) {
      row_data <- setNames(rep(NA_character_, length(cls)), cls)
    } else {
      row_data <- setNames(sapply(cls, function(cl){
        dsub <- df %>% filter(class==cl)
        summ_one_var(dsub[[v]], v)
      }), cls)
    }
    var_label <- if (!is.null(label_map[[v]])) label_map[[v]] else v
    row_df <- tibble(Variable = var_label) %>%
      bind_cols(as_tibble_row(row_data))
    all_rows[[length(all_rows) + 1]] <- row_df
  }
  bind_rows(all_rows)
}

# Group Table 2 variables into categories (similar to Table 1)
vars_table2 <- list(
  Demographics = c("Age_years", "Male_sex", "White_ethnicity"),
  Outcomes = c("ICU_mortality", "ICU_LOS_days", "Mortality_Day7", "ICU_Discharge_Day7", "Mortality_Day14", "ICU_Discharge_Day14"),
  `Organ Support` = c("Mechanical_ventilation", "NE_equiv_mcgkgmin_users"),
  `Respiratory Parameters` = c("RR_bpm", "PF_mmHg", "PaCO2_mmHg", "PEEP_cmH2O", "DrivingPressure_cmH2O", "MinuteVent_Lmin"),
  `Laboratory Values` = c("pH", "Lactate_mmolL", "Bicarbonate_mmolL", "Creatinine_mgdl", "WBC_10e9L", "Platelet_10e9L"),
  `Severity Scores` = c("SOFA_total", "Charlson_index", "Hyperinflammatory")
)

# Flatten for building table
vars_table2_flat <- unlist(vars_table2, use.names = FALSE)

# Filter to only variables present in each dataset
vars_mim  <- intersect(vars_table2_flat, names(base_mim))
vars_icht <- intersect(vars_table2_flat, names(base_icht))
vars_aumc <- intersect(vars_table2_flat, names(base_aumc))

# Build tables without categories first
t2_mim_base  <- build_by_class_table(base_mim,  vars_mim, label_map2)
t2_icht_base <- build_by_class_table(base_icht, vars_icht, label_map2)
t2_aumc_base <- build_by_class_table(base_aumc, vars_aumc, label_map2)

# Function to add category headers to Table 2
add_table2_categories <- function(tbl_data, df, vars_grouped, label_map) {
  cls <- levels(df$class)
  # Start with empty tibble with correct column structure
  result <- tibble(Variable = character())
  for (cl in cls) result[[cl]] <- character()
  
  # Add n and n (%) of cohort rows at the top
  n_rows <- tbl_data %>% filter(Variable %in% c("n", "n (%) of cohort"))
  result <- bind_rows(result, n_rows)
  
  for (cat_name in names(vars_grouped)) {
    # Add category header row
    cat_row <- tibble(Variable = cat_name)
    for (cl in cls) cat_row[[cl]] <- ""
    result <- bind_rows(result, cat_row)
    
    # Add variables in this category
    cat_vars <- vars_grouped[[cat_name]]
    cat_labels <- label_map[cat_vars]
    cat_data <- tbl_data %>% filter(Variable %in% cat_labels)
    result <- bind_rows(result, cat_data)
  }
  result
}

# Apply category headers to each table
t2_mim  <- add_table2_categories(t2_mim_base,  base_mim,  vars_table2, label_map2)
t2_icht <- add_table2_categories(t2_icht_base, base_icht, vars_table2, label_map2)
t2_aumc <- add_table2_categories(t2_aumc_base, base_aumc, vars_table2, label_map2)

# Helper function to identify category header rows and n rows
is_category_or_n_row <- function(df) {
  # Category rows have the Variable field matching category names
  # n rows are "n" and "n (%) of cohort"
  cat_names <- names(vars_table2)
  df$Variable %in% c(cat_names, "n", "n (%) of cohort")
}

ft2_mim  <- flextable(t2_mim)  %>% 
  fontsize(size = 8, part = "all") %>%  # Small font to fit many variables on page
  bold(i = ~ is_category_or_n_row(t2_mim), j = "Variable", bold = TRUE) %>%  # Bold category headers and n rows
  autofit() %>% 
  set_caption("Table 2 – MIMIC: Features by class (median [IQR]; first row n (% of cohort); no tests).")

ft2_icht <- flextable(t2_icht) %>% 
  fontsize(size = 8, part = "all") %>%  # Small font to fit many variables on page
  bold(i = ~ is_category_or_n_row(t2_icht), j = "Variable", bold = TRUE) %>%  # Bold category headers and n rows
  autofit() %>% 
  set_caption("Table 2 – ICHT: Features by class (median [IQR]; first row n (% of cohort); no tests).")

ft2_aumc <- flextable(t2_aumc) %>% 
  fontsize(size = 8, part = "all") %>%  # Small font to fit many variables on page
  bold(i = ~ is_category_or_n_row(t2_aumc), j = "Variable", bold = TRUE) %>%  # Bold category headers and n rows
  autofit() %>% 
  set_caption("Table 2 – AUMC: Features by class (median [IQR]; first row n (% of cohort); no tests).")

save_as_docx("Table 2 – MIMIC" = ft2_mim,  path = file.path(getwd(), "table2_mimic_by_class.docx"))
save_as_docx("Table 2 – ICHT"  = ft2_icht, path = file.path(getwd(), "table2_icht_by_class.docx"))
save_as_docx("Table 2 – AUMC"  = ft2_aumc, path = file.path(getwd(), "table2_aumc_by_class.docx"))

cat("Saved:\n",
    normalizePath("table1_baseline_day0.docx"), "\n",
    normalizePath("supp_vitals_mimic_icht.docx"), "\n",
    normalizePath("table2_mimic_by_class.docx"), "\n",
    normalizePath("table2_icht_by_class.docx"), "\n",
    normalizePath("table2_aumc_by_class.docx"), "\n")

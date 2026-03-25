# AHRF PF Trajectory — Second Submission Code

Analysis code for:
**"Reproducible Clinical Archetypes in Acute Respiratory Failure: A Multi-Cohort Trajectory Analysis"**

Three cohorts: MIMIC-IV (derivation, n=3,938), ICHT (external validation, n=2,888), AUMC (external validation, n=3,592).
ICHT not publicly available.
---

## Data Requirements

All scripts expect data files in a `Data/` subdirectory of the working directory (`AHRF-Trajectories/`). Patient-level data are not included in this repository. Pre-fitted CRLCMM model objects (`.rds`) are required in `cr_models/`.

---

## Script Overview

| Script | Purpose | Run? |
|--------|---------|------|
| `01_MIMIC_CRLCMM_fitting.R` | Fit Competing Risk Latent Class Mixed Model (CRLCMM) on MIMIC-IV | Pre-fit — not required |
| `02_AUMC_CRLCMM_fitting.R` | Fit CRLCMM de novo on AUMC | Pre-fit — not required |
| `03_ICHT_CRLCMM_denovo_and_comparison.R` | Fit CRLCMM de novo on ICHT + cross-cohort comparison | Pre-fit — not required |
| `04_CR_trajectory_class_assignment_comparison_figures.R` | Assign trajectory classes; generate empirical vs predicted PF and CIF figures for all 3 cohorts 
| `05_Class_Prediction_MIMIC_Train.R` | Train XGBoost class prediction models on MIMIC (days 0–7, binary + multiclass) | 
| `06_Class_Prediction_External_Validation_v4.R` | External validation of class prediction models in ICHT + AUMC; frozen + recalibrated; SPEC/T05/Youden thresholds |
| `06_Class_Prediction_External_Validation_v5.R` | As v4 but T05 threshold only (simpler output) |
| `07_Tables_for_manuscript.R` | Formatted docx tables for all 3 cohorts (baseline + class characteristics) | 
| `08_Added_Value_Analysis.R` | Prognostic added value of class prediction (ROC, DCA) |
| `09_ARDS_Sensitivity_Analysis.R` | ARDS vs non-ARDS subset sensitivity analysis in MIMIC |
| `10_Hyperinflammatory_AI_Classifier.R` | Apply external AI hyperinflammatory phenotype classifier to MIMIC + ICHT 
| `11_MIMIC_Hierarchical_Clustering.R` | Consensus hierarchical clustering (K=2–6) as supplementary comparison |
| `12_MIMIC_Longitudinal_Kmeans.R` | Longitudinal k-means as supplementary comparison |  Requires R ≤ 4.1 (`kml` package unavailable in R 4.5+) |
| `13_Consort_Diagram.R` | Generate CONSORT flow diagram (Figure 1) |

---

## Recommended Run Order

```
04 → 05 → 06v4 → 07 → 07b → 07c → 08 → 09 → 10 → 11 → 13
```

Scripts 04–13 are independent of 01–03 (they use pre-fitted model objects).
Script 06 depends on the trained models from script 05.

---

## Software Requirements

- **R ≥ 4.2** (tested on R 4.5.2)
- **xgboost ≥ 3.0** (scripts updated for 3.x API)
- Key packages: `lcmm`, `xgboost`, `caret`, `pROC`, `tableone`, `flextable`, `officer`, `ggplot2`, `DiagrammeR`, `DiagrammeRsvg`, `rsvg`, `pheatmap`, `cluster`, `cmprsk`, `survival`

Script 12 additionally requires `kml` and `longitudinalData`, which are only available for R ≤ 4.1.

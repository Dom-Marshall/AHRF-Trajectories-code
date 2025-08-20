Cohort and Class Characteristics for tables.R
- Generate cohort and class characteristics tables, requires process and cleaned data for MIMIC and ICHT


MIMIC_Competing_Risk_14 days_Quad_hpc.R
- Run competing risk LCMM models to generated 2-5 class models using MIMIC Cohort, also plot predicted trajectories and outcomes for K classes and summary plot for model analytics 
- All models are saved as RDS file "mimic_multiclass_joint_models_full_14d.rds" from which model parameters are extracted for subsequent class assignments

MIMIC Competing Risk 14 days Quad.R
- Final competing risk latent class mixed model - for chosen 4 class model, this can be selected and $best parameters are those used for main CRLCMM model
- Quadratic terms for PF
- Option to run on cluster (FORK) or locally (PSOCK, windows)

CR trajectory class assignment - comparison - figures.R
- Loads CRLCMM Model trained on MIMIC - selected 4 class model, loaded predicted PF and survival curves, compared with observed PF / outcome first for MIMIC-IV Cohort then ICHT Cohort

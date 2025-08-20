Cohort and Class Characteristics for tables.R
- Generate cohort and class characteristics tables, requires process and cleaned data for MIMIC and ICHT

MIMIC Competing Risk 14 days Quad.R
- Run competing risk LCMM models to generated 2-5 class models using MIMIC Cohort, also plot predicted trajectories and outcomes for K classes and summary plot for model analytics 
- Quadratic terms for PF
- Option to run on cluster (FORK) or locally (PSOCK, windows)
- All models are saved as RDS file "mimic_multiclass_joint_models_full_14d.rds" from which model parameters are extracted for subsequent class assignments

mimic_multiclass_joint_models_full_14d.rds
- Final competing risk latent class mixed model - for chosen 4 class model, this can be selected and $best parameters are those used for main CRLCMM model

CR trajectory class assignment - comparison - figures.R
- Loads CRLCMM Model trained on MIMIC - selected 4 class model, loaded predicted PF and survival curves, compared with observed PF / outcome first for MIMIC-IV Cohort then ICHT Cohort

Class Prediction Model MIMIC Train.R
- Generate Classification models multiclass and one vs other using 3 different feature sets generated from 2 static and 10 dynamic variables
- Also code to plot other variables by class

Class Prediction Model Validation - ICHT.R
- Testing MIMIC Trained model on ICHT data for classification of CRLCMM classes 
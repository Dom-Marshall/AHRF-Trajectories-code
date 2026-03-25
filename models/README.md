# Saved XGBoost Models

Pre-fitted XGBoost models produced by `06_Class_Prediction_External_Validation_v4.R`.
Models are saved in XGBoost JSON format (xgboost ≥ 1.6) and can be loaded with `xgb.load()`.

---

## Directory Structure

```
models/
├── frozen_MIMIC/
│   ├── multiclass/        # Softprob multiclass model trained on 80% MIMIC
│   │   ├── day0_multiclass.json
│   │   ├── day1_multiclass.json
│   │   └── ... (day0–day7)
│   └── binary_OvR/        # One-vs-rest binary models trained on 80% MIMIC
│       ├── day0_C1.json
│       ├── day0_C2.json
│       ├── day0_C3.json
│       ├── day0_C4.json
│       └── ... (day0–day7, one file per class)
├── recal_ICHT/
│   ├── multiclass/        # Softprob multiclass model recalibrated on 80% ICHT
│   └── binary_OvR/        # OvR binary models recalibrated on 80% ICHT
└── recal_AUMC/
    ├── multiclass/        # Softprob multiclass model recalibrated on 80% AUMC
    └── binary_OvR/        # OvR binary models recalibrated on 80% AUMC
```

---

## File Naming Convention

| Pattern | Description |
|---------|-------------|
| `dayD_multiclass.json` | Multiclass model built using features from days 0–D |
| `dayD_C1.json` | Binary OvR model for Class 1 (trajectory archetype 1) using features from days 0–D |
| `dayD_C2.json` | Binary OvR model for Class 2 |
| `dayD_C3.json` | Binary OvR model for Class 3 |
| `dayD_C4.json` | Binary OvR model for Class 4 |

D ranges from 0 (admission features only) to 7 (features accumulated over 8 days).

---

## How to Load

```r
library(xgboost)

# Load a multiclass model (frozen MIMIC, using features up to day 3)
mc_model <- xgb.load("models/frozen_MIMIC/multiclass/day3_multiclass.json")

# Load a binary OvR model (recalibrated ICHT, Class 2, day 3)
bin_c2 <- xgb.load("models/recal_ICHT/binary_OvR/day3_C2.json")

# Predict (requires a numeric feature matrix with the same column order as training)
# feature_matrix: a matrix with columns = predictor names used in training
pred_probs <- predict(mc_model, xgb.DMatrix(feature_matrix))
# For multiclass softprob, reshape:
n_classes <- 4
prob_matrix <- matrix(pred_probs, ncol = n_classes, byrow = TRUE)
# Columns correspond to C1, C2, C3, C4
```

---

## Feature Sets

Feature names and their order are determined by the `wide_mim()` / `wide_icht()` / `wide_aumc()`
helper functions in script `06_Class_Prediction_External_Validation_v4.R`.

The predictor set includes:
- **Static** (day 0 only): age, sex, admission SOFA components, APACHE-II score
- **Trajectory** (days 0–D): daily averages of PaO2/FiO2, PEEP, peak inspiratory pressure,
  minute volume, respiratory rate, PaCO2, lactate, creatinine, bicarbonate, noradrenaline dose

Day-0 models use only static features; later windows accumulate trajectory summaries.

---

## Reproducibility Note

Models were trained with a fixed random seed (`set.seed(1 + d)` for MIMIC, `2 + d` for ICHT,
`3 + d` for AUMC). Re-running script 06 will regenerate identical models provided the input
data and xgboost version are unchanged.

- xgboost version used: **≥ 3.0** (JSON format; not compatible with legacy `.model` binary format)
- R version: **≥ 4.2**

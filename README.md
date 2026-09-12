# QuantileML — Uncertainty Quantification for Volcanic Eruption Mass Estimation

A machine-learning workflow for estimating volcanic eruption mass using **Gradient Boosting Regression Trees (GBRT)**, **log-target transformation**, **quantile regression**, and **predictive uncertainty quantification**.

The workflow produces median mass predictions together with **P10–P90 uncertainty estimates**.

---

## Scientific Reference

This software is associated with the following publication:

**Mousavi, N., Fullea, J., & Mousavi, S. M. (2026).**
*A machine learning approach for volcanic eruption mass estimation.*
**Journal of Geophysical Research: Machine Learning and Computation, 3**, e2026JH001264.

**DOI:** https://doi.org/10.1029/2026JH001264

If you use this software in academic work, please cite the publication above.

Citation information is also provided in `CITATION.cff`.

---

## Overview

`QuantileML` estimates volcanic eruption mass from a set of input features using Gradient Boosting Regression Trees.

The target variable (`Mass`) is transformed into logarithmic space before model training. Three independent quantile regression models are then trained to estimate:

| Quantile | Interpretation            |
| -------- | ------------------------- |
| P10      | Lower prediction quantile |
| P50      | Median prediction         |
| P90      | Upper prediction quantile |

The P10–P90 interval provides an estimate of predictive uncertainty for each sample.

### Workflow

```text
Input Data
    │
    ▼
Feature Selection
    │
    ▼
Standardization
    │
    ▼
Log(Mass + EPS)
    │
    ▼
GBRT Quantile Regression
    │
    ├── P10
    ├── P50
    └── P90
    │
    ▼
Inverse Log Transformation
    │
    ▼
Bias Correction
    │
    ├── Median Mass
    ├── Lower Bound
    └── Upper Bound
    │
    ├───────────────┐
    ▼               ▼
CSV Output      Uncertainty Plot
```

---

## Model

The implementation uses `GradientBoostingRegressor` from Scikit-learn.

### GBRT Parameters

```python
GBRT_PARAMS = dict(
    loss="huber",
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=6,
    max_features=0.3,
    subsample=1.0,
    random_state=42,
)
```

For quantile regression, the loss function is changed to:

```python
loss="quantile"
```

with:

```python
QUANTILES = [0.1, 0.5, 0.9]
```

The three models therefore estimate the P10, P50, and P90 conditional prediction quantiles.

---

## Log-Target Transformation

Volcanic eruption masses can span several orders of magnitude. The model therefore operates in logarithmic target space.

The transformation is:

```python
y_log = np.log(y + EPS)
```

where:

```python
EPS = 1e-6
```

After prediction, the values are transformed back to the original mass scale:

```python
y_pred = np.exp(y_pred_log) - EPS
```

This approach helps accommodate the strongly skewed distribution and large dynamic range of volcanic eruption masses.

---

## Feature Standardization

Predictor variables are standardized using `StandardScaler`.

The scaler is fitted exclusively to the training data and then applied to the test data:

```python
scaler.fit_transform(X)
scaler.transform(X_test)
```

This prevents the test dataset from influencing the scaling parameters.

---

## Bias Correction

After obtaining the P50 prediction, a global multiplicative bias correction is applied:

```python
bias = np.mean(y) / np.mean(y_median)
```

The correction factor is then applied to all three quantiles:

```python
y_lower *= bias
y_median *= bias
y_upper *= bias
```

This correction aligns the overall mean prediction scale with the mean observed training mass.

---

# Repository Structure

```text
QuantileML_Uncertainty/
│
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── Uncertainty_Quantification.py
│
├── data/
│   ├── global.csv
│   └── gris_features.csv
│
├── plot/
│   └── errorbar_plot.png
│
├── prediction/
│   └── predicted_mass_with_uncertainty.csv
│
└── src/
    ├── __init__.py
    ├── load_data.py
    ├── plotting.py
    └── save_csv.py
```

### Main Components

**`Uncertainty_Quantification.py`**

Main entry point for the complete workflow, including model training, quantile prediction, bias correction, and execution of the analysis.

**`src/load_data.py`**

Contains functions related to reading and preprocessing the input datasets.

**`src/plotting.py`**

Contains visualization functions used to generate the uncertainty/error-bar plot.

**`src/save_csv.py`**

Contains functions for saving model predictions and uncertainty estimates.

**`data/`**

Contains the input datasets used by the workflow.

**`plot/`**

Contains generated figures.

**`prediction/`**

Contains generated prediction tables.

---

# Input Data

## Training Dataset

The training dataset is:

```text
data/global.csv
```

It must contain:

* `Mass` — observed volcanic eruption mass
* `lon` — longitude
* `lat` — latitude
* Predictor/feature columns

The feature columns are automatically selected by excluding:

```python
["Mass", "lon", "lat"]
```

The first CSV column is treated as the index.

---

## Test Dataset

The test dataset is:

```text
data/gris_features.csv
```

It must contain:

* The same predictor columns used for training
* `lon`
* `lat`

The test dataset does not require a `Mass` column.

The feature names and preprocessing requirements must be compatible between the training and test datasets.

---

# Installation

Python 3.9 or later is recommended.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
matplotlib
scikit-learn
```

---

# Usage

From the root directory of the repository, run:

```bash
python Uncertainty_Quantification.py
```

The program will:

1. Load the training and test datasets.
2. Select the model features.
3. Standardize the features.
4. Transform the target into logarithmic space.
5. Train the P10, P50, and P90 quantile models.
6. Generate predictions.
7. Apply bias correction.
8. Save the predictions and uncertainty estimates.
9. Generate an uncertainty visualization.

The program also prints a startup banner identifying the software and copyright information.

---

# Outputs

## Prediction CSV

The prediction file is generated at:

```text
prediction/predicted_mass_with_uncertainty.csv
```

The output contains:

| Column              | Description        |
| ------------------- | ------------------ |
| `lon`               | Longitude          |
| `lat`               | Latitude           |
| `mass_median`       | P50 predicted mass |
| `mass_lower`        | P10 predicted mass |
| `mass_upper`        | P90 predicted mass |
| `uncertainty_range` | P90 − P10          |

The mass unit is **Gt (gigatonnes)**, assuming the input `Mass` variable is provided in Gt.

---

## Uncertainty Plot

The generated figure is:

```text
plot/errorbar_plot.png
```

The plot displays:

* Red points — P50 predicted mass
* Grey error bars — P10–P90 predictive interval
* Sorted predictions along the x-axis
* Logarithmic mass axis
* Mean uncertainty reported in linear Gt

---

# Uncertainty Interpretation

The P10–P90 interval represents the range between the 10th and 90th conditional prediction quantiles estimated by the fitted quantile regression models.

Conceptually:

```text
P10 ───────────── P50 ───────────── P90
 │                 │                 │
Lower            Median             Upper
```

A wider P10–P90 interval indicates a broader predictive distribution, while a narrower interval indicates a more concentrated prediction.

The P10–P90 interval should **not automatically be interpreted as a classical statistical confidence interval**.

It represents predictive uncertainty estimated by the quantile models and does not necessarily capture every source of geological, observational, or model uncertainty.

---

# Reproducibility

The models use:

```python
random_state=42
```

to ensure reproducible behavior for the configured stochastic components.

For reproducible results, use consistent:

* Input datasets
* Feature definitions
* Python version
* Package versions
* Model parameters
* Preprocessing procedures

---

# Data and Scientific Considerations

The quality of predictions depends strongly on the quality and representativeness of the training data.

Particular care should be taken when applying the model to samples that are substantially outside the feature distribution represented by the training dataset.

Predictions should therefore be interpreted in the context of the training-data coverage and the scientific assumptions underlying the selected features.

---

# Citation

If you use this software or methodology in a publication, thesis, report, or other academic work, please cite:

```bibtex
@article{Mousavi2026VolcanicMass,
  author  = {Mousavi, N. and Fullea, J. and Mousavi, S. M.},
  title   = {A machine learning approach for volcanic eruption mass estimation},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  volume  = {3},
  pages   = {e2026JH001264},
  year    = {2026},
  doi     = {10.1029/2026JH001264}
}
```

The repository also contains a `CITATION.cff` file for machine-readable citation metadata.

---

# License

Copyright © 2026 **N. Mousavi, J. Fullea, and S. M. Mousavi**.

All rights reserved.

See the `LICENSE` file for the terms governing the use, reproduction, modification, and distribution of this software.

---

# Acknowledgments

This software was developed in support of research on machine-learning-based volcanic eruption mass estimation and predictive uncertainty quantification.

For the scientific methodology and detailed research context, please refer to the associated publication.

---

# Contact

For scientific questions, methodology-related issues, or questions concerning the implementation, please refer to the associated publication and the repository documentation.

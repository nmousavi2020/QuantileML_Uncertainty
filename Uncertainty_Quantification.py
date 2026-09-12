# =============================================================================
# Volcano Mass Uncertainty Estimation
# GBRT + Log-Target Transformation + Quantile Regression + Uncertainty
# =============================================================================
#
# Copyright (c) 2026 N. Mousavi, J. Fullea, and S. M. Mousavi
# All rights reserved.
#
# Description:
#     Machine-learning workflow for volcanic eruption mass estimation using
#     Gradient Boosting Regression Trees (GBRT), logarithmic target
#     transformation, quantile regression, and P10-P90 predictive uncertainty.
#
# Reference:
#     Mousavi, N., Fullea, J., & Mousavi, S. M. (2026).
#     A machine learning approach for volcanic eruption mass estimation.
#     Journal of Geophysical Research: Machine Learning and Computation, 3,
#     e2026JH001264.
#
# DOI:
#     https://doi.org/10.1029/2026JH001264
#
# Input files:
#     global.csv
#     gris_features.csv
#
# Outputs:
#     predictions/predicted_mass_with_uncertainty.csv
#     plots/errorbar_plot.png
#
# =============================================================================

print()
print("╭────────────────────────────────────────────────────────────────────╮")
print("│                 QuantileML — Uncertainty                           │")
print("│              GBRT + Log-Target Transformation                      │")
print("│           Quantile Regression + Predictive Uncertainty             │")
print("│                                                                    │")
print("│   Copyright © 2026 N. Mousavi, J. Fullea, and S. M. Mousavi        │")
print("│   For citation and reference information, please see the README.   │")
print("╰────────────────────────────────────────────────────────────────────╯")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from src.plotting import plot_errorbars
from src.save_csv import save_predictions_with_uncertainty
from src.load_data import load_data

print("\nAll libraries and functions were imported sucessfully!")
print(f"---------------------------------------------------------------------------------")

# ---------------------- CONFIG ----------------------
EPS = 1e-6
QUANTILES = [0.1, 0.5, 0.9]

TRAIN_FILE = "data/global.csv"
TEST_FILE = "data/gris_features.csv"

OUT_PLOT_DIR = "plot"
OUT_PRED_DIR = "prediction"

os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_PRED_DIR, exist_ok=True)

# ---------------------- MODEL setting ----------------------
GBRT_PARAMS = dict(

    loss="huber",
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=6,
    max_features=0.3,
    subsample=1.0,
    random_state=42,
)

QUANTILE_PARAMS = GBRT_PARAMS.copy()
QUANTILE_PARAMS.update({"loss": "quantile"})

# ---------------------- MAIN ----------------------
if __name__ == "__main__":

    X, y, X_test, coords_test = load_data(TRAIN_FILE,TEST_FILE)

    # Train quantile models
    y_log = np.log(y + EPS)
    quantile_preds = {}

    print("Here are the training quantiles:")

    for q in QUANTILES:
        print(f"Training quantile q={q}")
        model = GradientBoostingRegressor(**QUANTILE_PARAMS, alpha=q)
        model.fit(X, y_log)
        quantile_preds[q] = np.exp(model.predict(X_test)) - EPS

    y_lower = quantile_preds[0.1]
    y_median = quantile_preds[0.5]
    y_upper = quantile_preds[0.9]

    # Save predictions
    save_predictions_with_uncertainty(coords_test, y_median, y_lower, y_upper, OUT_PRED_DIR)

    # Plot error bars with mean uncertainty
    plot_errorbars(y_median, y_lower, y_upper, OUT_PLOT_DIR)

    print("\nThe uncertainty of prediction was computed sucessfully!")

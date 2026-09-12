import os

def save_predictions_with_uncertainty(coords, y_med, y_low, y_up, OUT_PRED_DIR):
    df = coords.copy()
    df["mass_median"] = y_med
    df["mass_lower"] = y_low
    df["mass_upper"] = y_up
    df["uncertainty_range"] = y_up - y_low

    path = os.path.join(OUT_PRED_DIR, "predicted_mass_with_uncertainty.csv")
    df.to_csv(path, index=False)
    print(f"---------------------------------------------------------------------------------")
    print(f"prediction is saved here: {path}")
    print(f"---------------------------------------------------------------------------------")
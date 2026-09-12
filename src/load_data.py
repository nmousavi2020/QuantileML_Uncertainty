import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(TRAIN_FILE, TEST_FILE):
    train = pd.read_csv(TRAIN_FILE, index_col=0)
    test = pd.read_csv(TEST_FILE, index_col=0)

    feature_cols = train.columns.drop(["Mass", "lon", "lat"])

    X = train[feature_cols].copy()
    y = train["Mass"].copy()
    coords_test = test[["lon", "lat"]].copy()
    X_test = test[feature_cols].copy()

    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    return X, y, X_test, coords_test
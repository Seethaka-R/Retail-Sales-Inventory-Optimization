# src/feature_engineering.py
# Creates time-series features for the forecasting model
# KEY CONCEPT: Lag features tell the model "what happened N days ago"

import os

import numpy as np
import pandas as pd


def create_features(
    input_path="data/processed/cleaned_data.csv",
    output_path="data/processed/featured_data.csv",
):
    """
    Adds lag features, rolling statistics, and time flags to cleaned data.
    These features are how a non-temporal model (Random Forest) learns time patterns.
    """
    print("=== FEATURE ENGINEERING STARTED ===")

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values(["product", "store", "date"]).reset_index(drop=True)

    group_cols = ["product", "store"]

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["units_sold"].shift(lag)

    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols)["units_sold"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(group_cols)["units_sold"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    df["rolling_max_30"] = (
        df.groupby(group_cols)["units_sold"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).max())
    )

    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features created: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(
        "Feature columns: "
        f"{[c for c in df.columns if c.startswith(('lag_', 'rolling_', 'is_', 'month_', 'weekday_'))]}"
    )
    print(f"Saved -> {output_path}")
    print("=== FEATURE ENGINEERING COMPLETE ===")
    return df


if __name__ == "__main__":
    create_features()

# src/forecasting_model.py
# Trains, evaluates, and saves the Random Forest forecasting model

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Feature columns used for training
FEATURE_COLS = [
    "store_code",
    "product_code",
    "category_code",
    "year",
    "month",
    "day",
    "weekday",
    "quarter",
    "week_of_year",
    "is_weekend",
    "is_promo",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_30",
    "rolling_max_30",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "is_month_start",
    "is_month_end",
    "is_quarter_end",
    "unit_price",
    "lead_time_days",
]

TARGET_COL = "units_sold"


def train_model(
    input_path="data/processed/featured_data.csv",
    model_path="models/rf_model.pkl",
    pred_path="data/outputs/predictions.csv",
):
    """
    Trains Random Forest Regressor on time-based train/test split.
    Time-based split (not random) is critical for time-series data.
    """
    print("=== MODEL TRAINING STARTED ===")

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Train: {len(train_df):,} rows ({train_df['date'].min()} -> {train_df['date'].max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['date'].min()} -> {test_df['date'].max()})")

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    X_train = train_df[available_features]
    y_train = train_df[TARGET_COL]
    X_test = test_df[available_features]
    y_test = test_df[TARGET_COL]

    print("Training Random Forest... (this may take ~2 minutes)")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        # Use a single process to avoid Windows joblib worker permission failures.
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

    print("\n--- MODEL EVALUATION ---")
    print(f"MAE  (Mean Absolute Error):        {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error):    {rmse:.2f}")
    print(f"R2   (Coefficient of Determination): {r2:.4f}")
    print(f"MAPE (Mean Abs Percentage Error):  {mape:.1f}%")

    results_df = test_df[["date", "product", "store", "category", "units_sold"]].copy()
    results_df["predicted_sales"] = y_pred.astype(int)
    results_df["residual"] = results_df["units_sold"] - results_df["predicted_sales"]

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    results_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved -> {pred_path}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "features": available_features}, model_path)
    print(f"Model saved -> {model_path}")

    feat_imp = pd.Series(model.feature_importances_, index=available_features)
    feat_imp = feat_imp.sort_values(ascending=False).head(15)

    os.makedirs("images/model", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("images/model/feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance chart saved.")

    sample = results_df[results_df["product"] == results_df["product"].unique()[0]]
    sample = sample[sample["store"] == "Store_A"].sort_values("date").head(90)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample["date"], sample["units_sold"], label="Actual", linewidth=1.5, color="#2E86AB")
    ax.plot(
        sample["date"],
        sample["predicted_sales"],
        label="Predicted",
        linewidth=1.5,
        linestyle="--",
        color="#E84855",
    )
    ax.set_title(f"Actual vs Predicted Sales - {sample['product'].iloc[0]}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("images/model/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Actual vs Predicted chart saved.")

    print("=== MODEL TRAINING COMPLETE ===")
    return model, results_df


if __name__ == "__main__":
    train_model()

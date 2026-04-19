# src/preprocessing.py
# Cleans and prepares the raw retail data for analysis and modeling

import os

import numpy as np
import pandas as pd


def load_and_clean(
    input_path="data/raw/retail_sales_data.csv",
    output_path="data/processed/cleaned_data.csv",
):
    """
    Loads raw retail data, performs thorough cleaning, and saves result.
    Returns cleaned DataFrame.
    """
    print("=== PREPROCESSING STARTED ===")

    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Null counts:\n{df.isnull().sum()}")

    df["date"] = pd.to_datetime(df["date"])
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")

    df["units_sold"] = df.groupby("product")["units_sold"].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"Nulls after fill: {df['units_sold'].isnull().sum()}")

    q1 = df["units_sold"].quantile(0.25)
    q3 = df["units_sold"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    before = len(df)
    df = df[(df["units_sold"] >= lower) & (df["units_sold"] <= upper)]
    print(f"Outlier rows removed: {before - len(df)}")

    df = df[df["units_sold"] >= 0]
    df["units_sold"] = df["units_sold"].astype(int)

    df["store_code"] = df["store"].astype("category").cat.codes
    df["category_code"] = df["category"].astype("category").cat.codes
    df["product_code"] = df["product"].astype("category").cat.codes

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df = df.sort_values(["product", "store", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved: {df.shape[0]:,} rows -> {output_path}")
    print("=== PREPROCESSING COMPLETE ===")
    return df


if __name__ == "__main__":
    load_and_clean()

# src/data_generator.py
# Generates a realistic synthetic retail sales dataset
# Run this first to create your raw data

import os
from datetime import datetime

import numpy as np
import pandas as pd


def generate_retail_data(output_path="data/raw/retail_sales_data.csv"):
    """
    Generates 3 years of synthetic daily retail sales data.
    Simulates real patterns: seasonality, promotions, weekday effects.
    """
    np.random.seed(42)

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    products = {
        "Biscuits": {"category": "Food", "base_price": 30, "base_demand": 120},
        "Chips": {"category": "Food", "base_price": 20, "base_demand": 90},
        "Soft Drinks": {"category": "Beverage", "base_price": 40, "base_demand": 200},
        "Juice": {"category": "Beverage", "base_price": 60, "base_demand": 80},
        "Shampoo": {"category": "Personal Care", "base_price": 150, "base_demand": 40},
        "Soap": {"category": "Personal Care", "base_price": 50, "base_demand": 70},
        "Rice (5kg)": {"category": "Grocery", "base_price": 280, "base_demand": 60},
        "Cooking Oil": {"category": "Grocery", "base_price": 120, "base_demand": 55},
        "Detergent": {"category": "Household", "base_price": 90, "base_demand": 45},
        "Floor Cleaner": {"category": "Household", "base_price": 70, "base_demand": 35},
    }

    stores = ["Store_A", "Store_B", "Store_C"]
    records = []

    for date in dates:
        month = date.month
        weekday = date.dayofweek
        is_weekend = 1 if weekday >= 5 else 0

        if month in [10, 11, 12]:
            seasonal_factor = 1.4
        elif month in [6, 7, 8]:
            seasonal_factor = 0.85
        else:
            seasonal_factor = 1.0

        weekend_factor = 1.25 if is_weekend else 1.0
        is_promo = 1 if np.random.random() < 0.15 else 0
        promo_factor = 1.35 if is_promo else 1.0

        for product_name, info in products.items():
            for store in stores:
                base = info["base_demand"]
                noise = np.random.normal(1.0, 0.12)
                demand = int(base * seasonal_factor * weekend_factor * promo_factor * noise)
                demand = max(0, demand)

                stock_level = np.random.randint(50, 500)
                lead_time = np.random.choice([3, 5, 7])

                records.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store": store,
                        "product": product_name,
                        "category": info["category"],
                        "units_sold": demand,
                        "unit_price": info["base_price"],
                        "revenue": demand * info["base_price"],
                        "stock_level": stock_level,
                        "lead_time_days": lead_time,
                        "is_promo": is_promo,
                        "is_weekend": is_weekend,
                        "month": month,
                        "weekday": weekday,
                    }
                )

    df = pd.DataFrame(records)

    mask = np.random.random(len(df)) < 0.02
    df.loc[mask, "units_sold"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {len(df):,} rows -> {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Products: {df['product'].nunique()}, Stores: {df['store'].nunique()}")
    return df


if __name__ == "__main__":
    generate_retail_data()

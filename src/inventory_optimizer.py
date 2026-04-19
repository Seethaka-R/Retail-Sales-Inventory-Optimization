# src/inventory_optimizer.py
# Core inventory optimization logic
#
# KEY FORMULAS:
#   Reorder Point (ROP) = Average Daily Demand * Lead Time + Safety Stock
#   Safety Stock = Z * sigma_demand * sqrt(Lead Time)
#   EOQ (Economic Order Qty) = sqrt(2 * Annual Demand * Order Cost / Holding Cost)
#
# WHERE:
#   Z = service level factor (1.28 = 90%, 1.65 = 95%, 2.05 = 98%)
#   sigma_demand = standard deviation of daily demand
#   Lead Time = days between placing order and receiving stock

import os

import numpy as np
import pandas as pd

# Service level -> Z-score mapping
SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,
    0.98: 2.05,
    0.99: 2.33,
}


def calculate_inventory_metrics(
    pred_path="data/outputs/predictions.csv",
    cleaned_path="data/processed/cleaned_data.csv",
    output_path="data/outputs/inventory_recommendations.csv",
    service_level=0.95,
    order_cost=500,
    holding_cost_pct=0.25,
):
    """
    Calculates reorder points, safety stock, EOQ, and generates
    reorder alerts for each product-store combination.
    """
    print("=== INVENTORY OPTIMIZATION STARTED ===")

    z_value = SERVICE_LEVEL_Z.get(service_level, 1.65)
    print(f"Service level: {service_level*100:.0f}% (Z = {z_value})")

    preds = pd.read_csv(pred_path, parse_dates=["date"])
    cleaned = pd.read_csv(cleaned_path, parse_dates=["date"])

    price_lead = (
        cleaned.groupby(["product", "store"])
        .agg(unit_price=("unit_price", "mean"), lead_time_days=("lead_time_days", "mean"))
        .reset_index()
    )

    recommendations = []

    for (product, store), group in preds.groupby(["product", "store"]):
        daily_demand_mean = group["predicted_sales"].mean()
        daily_demand_std = group["predicted_sales"].std()

        pl_row = price_lead[(price_lead["product"] == product) & (price_lead["store"] == store)]
        if pl_row.empty:
            continue

        lead_time = pl_row["lead_time_days"].values[0]
        unit_price = pl_row["unit_price"].values[0]

        safety_stock = z_value * daily_demand_std * np.sqrt(lead_time)
        safety_stock = max(0, round(safety_stock))

        reorder_point = (daily_demand_mean * lead_time) + safety_stock
        reorder_point = round(reorder_point)

        annual_demand = daily_demand_mean * 365
        holding_cost = unit_price * holding_cost_pct

        if holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        else:
            eoq = annual_demand / 12
        eoq = max(1, round(eoq))

        latest_slice = cleaned[(cleaned["product"] == product) & (cleaned["store"] == store)]
        latest_stock = latest_slice["stock_level"].iloc[-1] if not latest_slice.empty else 100

        days_of_stock = (latest_stock / daily_demand_mean) if daily_demand_mean > 0 else 999

        needs_reorder = latest_stock <= reorder_point
        urgency = "CRITICAL" if days_of_stock < lead_time else ("REORDER NOW" if needs_reorder else "OK")

        forecast_30d = group.sort_values("date").tail(30)["predicted_sales"].sum()

        recommendations.append(
            {
                "product": product,
                "store": store,
                "avg_daily_demand": round(daily_demand_mean, 1),
                "demand_std_dev": round(daily_demand_std, 1),
                "lead_time_days": int(lead_time),
                "safety_stock_units": safety_stock,
                "reorder_point_units": reorder_point,
                "eoq_units": eoq,
                "current_stock": int(latest_stock),
                "days_of_stock_left": round(days_of_stock, 1),
                "reorder_needed": needs_reorder,
                "urgency_flag": urgency,
                "forecast_next_30d": int(forecast_30d),
                "unit_price_rs": unit_price,
                "reorder_cost_rs": round(eoq * unit_price, 0),
            }
        )

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values("days_of_stock_left").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rec_df.to_csv(output_path, index=False)

    critical = rec_df[rec_df["urgency_flag"] == "CRITICAL"]
    reorder = rec_df[rec_df["urgency_flag"] == "REORDER NOW"]

    print("\n--- INVENTORY SUMMARY ---")
    print(f"Total product-store combinations: {len(rec_df)}")
    print(f"CRITICAL (stock out within lead time): {len(critical)}")
    print(f"REORDER NOW: {len(reorder)}")
    print(f"OK: {len(rec_df) - len(critical) - len(reorder)}")
    print(f"\nRecommendations saved -> {output_path}")
    print("=== INVENTORY OPTIMIZATION COMPLETE ===")

    return rec_df


if __name__ == "__main__":
    calculate_inventory_metrics()

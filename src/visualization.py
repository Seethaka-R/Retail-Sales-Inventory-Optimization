# src/visualization.py
# Generates all charts for EDA, model results, and inventory

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Chart style settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = '#F8F9FA'
plt.rcParams['axes.grid']        = True
plt.rcParams['grid.alpha']       = 0.3
plt.rcParams['font.family']      = 'DejaVu Sans'

def run_eda(data_path="data/processed/cleaned_data.csv"):
    """Generates all EDA charts and saves to images/eda/"""
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    os.makedirs("images/eda", exist_ok=True)
    
    print("Generating EDA charts...")
    
    # --- 1. Overall Sales Trend ---
    daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
    daily_sales['rolling_7d'] = daily_sales['units_sold'].rolling(7).mean()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(daily_sales['date'], daily_sales['units_sold'], alpha=0.25, color='#2E86AB')
    ax.plot(daily_sales['date'], daily_sales['rolling_7d'], color='#2E86AB', linewidth=2, label='7-Day Rolling Avg')
    ax.set_title('Overall Daily Sales Trend (2021–2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Total Units Sold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/eda/sales_trend.png", dpi=150)
    plt.close()
    print("  ✓ sales_trend.png")
    
    # --- 2. Category-wise Sales ---
    cat_sales = df.groupby('category')['revenue'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#9B5DE5'][:len(cat_sales)]
    cat_sales.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Revenue (Rs.)')
    for i, v in enumerate(cat_sales.values):
        ax.text(v + cat_sales.max()*0.01, i, f'₹{v/1e6:.1f}M', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("images/eda/category_sales.png", dpi=150)
    plt.close()
    print("  ✓ category_sales.png")
    
    # --- 3. Monthly Seasonality ---
    monthly = df.groupby('month')['units_sold'].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(month_names, monthly.values,
                  color=['#E84855' if m in [10,11,12] else '#2E86AB' for m in range(1,13)])
    ax.set_title('Average Daily Sales by Month (Seasonality)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg Units Sold per Day')
    ax.axhline(monthly.mean(), color='#F4A261', linewidth=2, linestyle='--', label='Annual Average')
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/eda/monthly_seasonality.png", dpi=150)
    plt.close()
    print("  ✓ monthly_seasonality.png")
    
    # --- 4. Top 10 Products by Sales ---
    top_products = df.groupby('product')['units_sold'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.plot(kind='bar', ax=ax, color='#3BB273')
    ax.set_title('Top 10 Products by Total Units Sold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Units Sold')
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    plt.savefig("images/eda/top_products.png", dpi=150)
    plt.close()
    print("  ✓ top_products.png")
    
    # --- 5. Promo vs Non-Promo Sales ---
    promo_comp = df.groupby('is_promo')['units_sold'].mean()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Regular Day', 'Promotion Day']
    colors = ['#2E86AB', '#E84855']
    bars = ax.bar(labels, promo_comp.values, color=colors, width=0.5)
    for bar, val in zip(bars, promo_comp.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_title('Average Sales: Promo vs Non-Promo Days', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg Units Sold')
    uplift = ((promo_comp[1] - promo_comp[0]) / promo_comp[0]) * 100
    ax.text(0.5, 0.9, f'Promo Uplift: +{uplift:.1f}%', transform=ax.transAxes,
            ha='center', fontsize=12, color='green', fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/eda/promo_vs_regular.png", dpi=150)
    plt.close()
    print("  ✓ promo_vs_regular.png")
    
    # --- 6. Correlation Heatmap ---
    num_cols = ['units_sold','unit_price','stock_level','lead_time_days',
                'is_promo','is_weekend','month','weekday']
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/eda/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  ✓ correlation_heatmap.png")
    
    # --- 7. Store-wise Sales Comparison ---
    store_monthly = df.groupby(['month','store'])['units_sold'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for store, color in zip(['Store_A','Store_B','Store_C'], ['#2E86AB','#E84855','#3BB273']):
        data = store_monthly[store_monthly['store'] == store]
        ax.plot(data['month'], data['units_sold'], label=store, marker='o', color=color, linewidth=2)
    ax.set_title('Monthly Sales Comparison by Store', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Units Sold')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/eda/store_comparison.png", dpi=150)
    plt.close()
    print("  ✓ store_comparison.png")
    
    print("All EDA charts saved to images/eda/")

def plot_inventory_alerts(rec_path="data/outputs/inventory_recommendations.csv"):
    """Creates inventory status visualization"""
    
    df = pd.read_csv(rec_path)
    os.makedirs("images/inventory", exist_ok=True)
    
    # --- Days of stock remaining by product-store ---
    critical = df[df['urgency_flag'] == 'CRITICAL']
    reorder  = df[df['urgency_flag'] == 'REORDER NOW']
    ok       = df[df['urgency_flag'] == 'OK']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart: status breakdown
    ax = axes[0]
    sizes  = [len(critical), len(reorder), len(ok)]
    labels = ['CRITICAL', 'REORDER NOW', 'OK']
    colors = ['#E84855', '#F4A261', '#3BB273']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 12})
    ax.set_title('Inventory Status Distribution', fontsize=13, fontweight='bold')
    
    # Bar chart: top 10 most urgent by days of stock
    ax2 = axes[1]
    top_urgent = df.nsmallest(10, 'days_of_stock_left')
    bar_colors = ['#E84855' if f == 'CRITICAL' else '#F4A261' if f == 'REORDER NOW' else '#3BB273'
                  for f in top_urgent['urgency_flag']]
    ax2.barh(top_urgent['product'] + ' - ' + top_urgent['store'],
             top_urgent['days_of_stock_left'], color=bar_colors)
    ax2.set_xlabel('Days of Stock Remaining')
    ax2.set_title('Top 10 Most Urgent Reorder Items', fontsize=13, fontweight='bold')
    ax2.axvline(7, color='red', linestyle='--', alpha=0.7, label='7-day warning')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("images/inventory/reorder_alerts.png", dpi=150)
    plt.close()
    print("Inventory alerts chart saved → images/inventory/reorder_alerts.png")

if __name__ == "__main__":
    run_eda()
    plot_inventory_alerts()
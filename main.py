# main.py
# Run the complete pipeline: generate → clean → engineer → train → optimize → visualize
# Command: python main.py

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator       import generate_retail_data
from src.preprocessing        import load_and_clean
from src.feature_engineering  import create_features
from src.forecasting_model    import train_model
from src.inventory_optimizer  import calculate_inventory_metrics
from src.visualization        import run_eda, plot_inventory_alerts

def main():
    print("\n" + "="*60)
    print("  RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION SYSTEM")
    print("="*60 + "\n")
    
    print("[1/6] Generating synthetic retail dataset...")
    generate_retail_data()
    
    print("\n[2/6] Preprocessing and cleaning data...")
    load_and_clean()
    
    print("\n[3/6] Running exploratory data analysis...")
    run_eda()
    
    print("\n[4/6] Engineering features...")
    create_features()
    
    print("\n[5/6] Training forecasting model...")
    train_model()
    
    print("\n[6/6] Running inventory optimization...")
    calculate_inventory_metrics()
    plot_inventory_alerts()
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("  - Charts saved to: images/")
    print("  - Predictions:     data/outputs/predictions.csv")
    print("  - Inventory recs:  data/outputs/inventory_recommendations.csv")
    print("  - Model saved:     models/rf_model.pkl")
    print("\n  To launch the dashboard:")
    print("  python app/taipy_app.py")
    print("  Then open: http://localhost:8050")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
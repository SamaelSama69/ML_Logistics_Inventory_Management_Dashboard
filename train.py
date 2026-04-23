"""
train.py — Run this ONCE before starting the API to generate all model artifacts.

Usage (from the project root):
    python train.py

Outputs written to models/:
    demand_xgb.pkl
    demand_sku_encoder.pkl
    route_cost_lgbm.pkl
    route_cost_vehicle_encoder.pkl
    iso_forest.pkl
    if_scaler.pkl
    lstm_autoencoder.keras
    lstm_scaler.pkl
    lstm_threshold.pkl
"""

import pandas as pd
import sys
import os

# Make sure modules/ is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modules.demand_forecast  import DemandForecaster
from modules.route_cost       import RouteCostPredictor
from modules.predictive_maint import PredictiveMaintenanceSystem


# ── 1. Demand Forecaster ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("1/3  Training Demand Forecaster (XGBoost)...")
print("="*60)

demand_df = pd.read_csv('data/demand_data.csv')
print(f"    Loaded demand_data.csv  →  {len(demand_df):,} rows, "
      f"{demand_df['sku_id'].nunique()} SKUs")

# Train on a single representative SKU first to keep training fast,
# then retrain on the full dataset for production accuracy.
# For a quick smoke-test, set QUICK_TRAIN=1 in your environment.
if os.getenv('QUICK_TRAIN') == '1':
    sku_sample = demand_df['sku_id'].unique()[:10]
    train_df   = demand_df[demand_df['sku_id'].isin(sku_sample)]
    print(f"    QUICK_TRAIN: using {len(sku_sample)} SKUs only")
else:
    train_df = demand_df

forecaster = DemandForecaster()
mape = forecaster.fit(train_df)
print(f"    ✓  Demand model saved  |  MAPE: {mape:.2%}")


# ── 2. Route Cost Predictor ───────────────────────────────────────────────────
print("\n" + "="*60)
print("2/3  Training Route Cost Predictor (LightGBM)...")
print("="*60)

route_df = pd.read_csv('data/route_cost_data.csv')
print(f"    Loaded route_cost_data.csv  →  {len(route_df):,} rows")

predictor = RouteCostPredictor()
metrics   = predictor.fit(route_df)
print(f"    ✓  Route cost model saved  |  MAE: ₹{metrics['mae']:,.0f}  R²: {metrics['r2']:.3f}")


# ── 3. Predictive Maintenance ─────────────────────────────────────────────────
print("\n" + "="*60)
print("3/3  Training Predictive Maintenance System...")
print("="*60)

sensor_df = pd.read_csv('data/sensor_data.csv')
print(f"    Loaded sensor_data.csv  →  {len(sensor_df):,} rows  "
      f"({sensor_df['is_anomaly'].sum():,} labelled anomalies)")

pms = PredictiveMaintenanceSystem(contamination=0.05, sequence_length=30)

print("\n  [3a] Isolation Forest...")
pms.fit_isolation_forest(sensor_df)

print("\n  [3b] LSTM Autoencoder (normal rows only)...")
if os.getenv('QUICK_TRAIN') == '1':
    # Use one vehicle to keep LSTM training under a minute
    vehicle_sample = sensor_df['vehicle_id'].unique()[:3]
    lstm_df = sensor_df[sensor_df['vehicle_id'].isin(vehicle_sample)]
    print(f"    QUICK_TRAIN: using {len(vehicle_sample)} vehicles only")
else:
    lstm_df = sensor_df

pms.fit_lstm(lstm_df)
print(f"    ✓  LSTM saved  |  Anomaly threshold: {pms.threshold:.6f}")


# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("All models trained and saved to models/")
print("="*60)
print("\nArtifacts:")
for f in sorted(os.listdir('models')):
    size = os.path.getsize(f'models/{f}') / 1024
    print(f"    models/{f:<45}  {size:>8.1f} KB")

print("\nYou can now run the API:")
print("    uvicorn api.main:app --host 0.0.0.0 --port 8000")
print("  or")
print("    docker compose up --build")

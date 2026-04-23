import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ── DATASET NOTE ─────────────────────────────────────────────────────────────
# route_cost_data.csv columns match this module exactly:
#   distance_km, load_weight_kg, num_stops, fuel_price, vehicle_type,
#   departure_hour, driver_overtime_hrs, fuel_consumption_per_km,
#   vehicle_capacity_kg, base_cost, actual_route_cost
# No column renames needed.

REQUIRED_PREDICT_KEYS = [
    'distance_km', 'load_weight_kg', 'num_stops', 'fuel_price',
    'base_cost', 'vehicle_capacity_kg', 'fuel_consumption_per_km',
    'departure_hour', 'vehicle_type', 'driver_overtime_hrs',
]


class RouteCostPredictor:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            feature_fraction=0.8,
            random_state=42
        )
        self.vehicle_encoder = LabelEncoder()
        self._is_fitted = False

    def engineer_features(self, df, is_training=False):
        df = df.copy()

        if 'vehicle_type' in df.columns:
            if is_training:
                df['vehicle_type_encoded'] = self.vehicle_encoder.fit_transform(df['vehicle_type'])
            else:
                df['vehicle_type_encoded'] = self.vehicle_encoder.transform(df['vehicle_type'])

        safe_distance = df['distance_km'].clip(lower=1)
        safe_capacity = df['vehicle_capacity_kg'].clip(lower=1)

        df['cost_per_km']        = df['base_cost'] / safe_distance
        df['load_efficiency']    = df['load_weight_kg'] / safe_capacity
        df['stops_per_100km']    = (df['num_stops'] / safe_distance) * 100
        df['fuel_cost_estimate'] = (df['distance_km']
                                    * df['fuel_price']
                                    * df['fuel_consumption_per_km'])

        df['is_overnight']    = (
            (df['departure_hour'] >= 20) | (df['departure_hour'] <= 5)
        ).astype(int)
        df['is_peak_traffic'] = (
            df['departure_hour'].between(8, 10) | df['departure_hour'].between(17, 19)
        ).astype(int)

        return df

    def _feature_cols(self):
        return [
            'distance_km', 'load_weight_kg', 'num_stops', 'fuel_price',
            'cost_per_km', 'load_efficiency', 'stops_per_100km',
            'fuel_cost_estimate', 'is_overnight', 'is_peak_traffic',
            'vehicle_type_encoded', 'driver_overtime_hrs',
        ]

    def fit(self, df):
        os.makedirs('models', exist_ok=True)

        df = self.engineer_features(df, is_training=True)

        X = df[self._feature_cols()]
        y = df['actual_route_cost']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        preds = self.model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        print(f'MAE: ₹{mae:,.0f}')
        print(f'R²:  {r2:.3f}')

        joblib.dump(self.model,           'models/route_cost_lgbm.pkl')
        joblib.dump(self.vehicle_encoder, 'models/route_cost_vehicle_encoder.pkl')

        self._is_fitted = True
        return {'mae': mae, 'r2': r2}

    def load_model(self, model_path='models/route_cost_lgbm.pkl',
                   encoder_path='models/route_cost_vehicle_encoder.pkl'):
        self.model           = joblib.load(model_path)
        self.vehicle_encoder = joblib.load(encoder_path)
        self._is_fitted      = True
        print(f'Model loaded from {model_path}')

    def predict(self, route_features: dict) -> float:
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() or load_model() first."
            )

        missing = [k for k in REQUIRED_PREDICT_KEYS if k not in route_features]
        if missing:
            raise KeyError(f"predict() missing required keys: {missing}")

        df = pd.DataFrame([route_features])
        df = self.engineer_features(df, is_training=False)

        return float(self.model.predict(df[self._feature_cols()])[0])

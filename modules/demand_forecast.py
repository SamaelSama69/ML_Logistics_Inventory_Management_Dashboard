import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    def __init__(self):
        self.xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.sku_encoder = LabelEncoder()
        self._is_fitted = False

    def create_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        if 'sku_id' in df.columns:
            df['sku_encoded'] = self.sku_encoder.fit_transform(df['sku_id'])

        # Lag features
        df['lag_7']  = df['demand'].shift(7)
        df['lag_14'] = df['demand'].shift(14)
        df['lag_30'] = df['demand'].shift(30)

        # Rolling statistics
        df['rolling_mean_7']  = df['demand'].shift(1).rolling(7).mean()
        df['rolling_std_7']   = df['demand'].shift(1).rolling(7).std()
        df['rolling_mean_30'] = df['demand'].shift(1).rolling(30).mean()

        # Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month']       = df['date'].dt.month
        df['quarter']     = df['date'].dt.quarter
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['year']        = df['date'].dt.year

        # ── DATASET FIX ──────────────────────────────────────────────────────
        # demand_data.csv uses 'unit_price', not 'price'.
        # Rename so _get_feature_cols() can find it consistently.
        if 'unit_price' in df.columns and 'price' not in df.columns:
            df = df.rename(columns={'unit_price': 'price'})

        return df.dropna().reset_index(drop=True)

    def _get_feature_cols(self, df):
        """Return feature columns present in the given DataFrame."""
        base_cols = [
            'lag_7', 'lag_14', 'lag_30',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30',
            'day_of_week', 'month', 'quarter', 'is_weekend', 'year',
            'price',        # mapped from unit_price in create_features()
            'promotion',
        ]
        if 'sku_encoded' in df.columns:
            base_cols.append('sku_encoded')
        # Only keep cols that actually exist (graceful degradation)
        return [c for c in base_cols if c in df.columns]

    def fit(self, df):
        os.makedirs('models', exist_ok=True)

        df = self.create_features(df)
        feature_cols = self._get_feature_cols(df)

        X = df[feature_cols]
        y = df['demand']

        split = int(len(df) * 0.8)
        self.xgb_model.fit(X.iloc[:split], y.iloc[:split])

        preds = self.xgb_model.predict(X.iloc[split:])
        mape  = mean_absolute_percentage_error(y.iloc[split:], preds)
        print(f'XGBoost MAPE: {mape:.2%}')

        joblib.dump(self.xgb_model,   'models/demand_xgb.pkl')
        joblib.dump(self.sku_encoder, 'models/demand_sku_encoder.pkl')
        self._is_fitted = True
        return mape

    def load_model(self, path='models/demand_xgb.pkl',
                   encoder_path='models/demand_sku_encoder.pkl'):
        self.xgb_model   = joblib.load(path)
        # encoder is optional — may not exist if training was SKU-less
        if os.path.exists(encoder_path):
            self.sku_encoder = joblib.load(encoder_path)
        self._is_fitted = True
        print(f'Model loaded from {path}')

    def arima_forecast(self, series, steps=30):
        model = ARIMA(series, order=(2, 1, 2))
        fit   = model.fit()
        fc    = fit.get_forecast(steps=steps)
        return fc.predicted_mean, fc.conf_int()

    def predict(self, df, horizon=30):
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load_model() first."
            )

        df = self.create_features(df)
        feature_cols = self._get_feature_cols(df)

        xgb_pred = self.xgb_model.predict(df.tail(horizon)[feature_cols])
        arima_fc, conf = self.arima_forecast(df['demand'], steps=horizon)

        hybrid = (xgb_pred + arima_fc.values) / 2

        return {
            'forecast': hybrid.tolist(),
            'confidence_interval': conf.values.tolist(),
        }

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ── Keras import fix for TensorFlow 2.16+ ────────────────────────────────────
# TF 2.16 moved Keras into a standalone package (keras 3.x).
# The old path `from tensorflow.keras import ...` no longer works on Python 3.12.
# Use `import keras` directly — works with keras 3.x installed alongside TF 2.18.
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping

SENSOR_FEATURES = ['temperature', 'vibration', 'pressure', 'rpm', 'oil_level']


class PredictiveMaintenanceSystem:

    def __init__(self, contamination=0.05, sequence_length=30):
        self.contamination    = contamination
        self.sequence_length  = sequence_length
        self.if_scaler        = StandardScaler()
        self.lstm_scaler      = StandardScaler()
        self.iso_forest       = IsolationForest(
            contamination=contamination, n_estimators=200, random_state=42
        )
        self.lstm_model  = None
        self.threshold   = None
        self._if_fitted  = False
        self._lstm_fitted = False

    @staticmethod
    def _clean_sensor_df(sensor_df):
        drop_cols = [c for c in ['vehicle_id', 'timestamp', 'is_anomaly']
                     if c in sensor_df.columns]
        return sensor_df.drop(columns=drop_cols)

    # ── Isolation Forest ──────────────────────────────────────────────────────

    def fit_isolation_forest(self, sensor_df):
        os.makedirs('models', exist_ok=True)
        sensor_df = self._clean_sensor_df(sensor_df)
        X = self.if_scaler.fit_transform(sensor_df[SENSOR_FEATURES])
        self.iso_forest.fit(X)
        joblib.dump(self.iso_forest, 'models/iso_forest.pkl')
        joblib.dump(self.if_scaler,  'models/if_scaler.pkl')
        self._if_fitted = True
        print('Isolation Forest fitted and saved.')

    def detect_anomaly(self, sensor_reading: dict) -> dict:
        if not self._if_fitted:
            raise RuntimeError("Isolation Forest not fitted. Call fit_isolation_forest() or load_models() first.")
        X = self.if_scaler.transform([[
            sensor_reading['temperature'], sensor_reading['vibration'],
            sensor_reading['pressure'],    sensor_reading['rpm'],
            sensor_reading['oil_level'],
        ]])
        score      = self.iso_forest.decision_function(X)[0]
        is_anomaly = self.iso_forest.predict(X)[0] == -1
        return {'anomaly': bool(is_anomaly), 'anomaly_score': float(score)}

    # ── LSTM Autoencoder ──────────────────────────────────────────────────────

    def build_lstm_autoencoder(self, n_features):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, n_features), return_sequences=False),
            RepeatVector(self.sequence_length),
            LSTM(64, return_sequences=True),
            TimeDistributed(Dense(n_features)),
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _make_sequences(self, X):
        windows = np.lib.stride_tricks.sliding_window_view(X, (self.sequence_length, X.shape[1]))
        return windows[:, 0, :, :]

    def fit_lstm(self, sensor_df):
        os.makedirs('models', exist_ok=True)
        if 'is_anomaly' in sensor_df.columns:
            normal_df = sensor_df[sensor_df['is_anomaly'] == 0].copy()
            print(f'LSTM training on {len(normal_df):,} normal rows '
                  f'(dropped {len(sensor_df)-len(normal_df):,} anomaly rows)')
        else:
            normal_df = sensor_df.copy()
        normal_df = self._clean_sensor_df(normal_df)
        X         = self.lstm_scaler.fit_transform(normal_df[SENSOR_FEATURES])
        sequences = self._make_sequences(X)
        model     = self.build_lstm_autoencoder(len(SENSOR_FEATURES))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(sequences, sequences, epochs=50, batch_size=32,
                  validation_split=0.1, callbacks=[early_stop], verbose=1)
        model.save('models/lstm_autoencoder.keras')
        joblib.dump(self.lstm_scaler, 'models/lstm_scaler.pkl')
        preds          = model.predict(sequences, verbose=0)
        mse            = np.mean(np.power(sequences - preds, 2), axis=(1, 2))
        self.threshold = float(np.percentile(mse, 95))
        joblib.dump(self.threshold, 'models/lstm_threshold.pkl')
        self.lstm_model   = model
        self._lstm_fitted = True
        print(f'LSTM Autoencoder fitted. Threshold (95th pct MSE): {self.threshold:.6f}')

    def detect_anomaly_lstm(self, sensor_df) -> dict:
        if not self._lstm_fitted:
            raise RuntimeError("LSTM not fitted. Call fit_lstm() or load_models() first.")
        clean = self._clean_sensor_df(sensor_df)
        X     = self.lstm_scaler.transform(clean[SENSOR_FEATURES].values)
        seq   = X[-self.sequence_length:][np.newaxis, :, :]
        pred  = self.lstm_model.predict(seq, verbose=0)
        error = float(np.mean(np.power(seq - pred, 2)))
        return {'anomaly': error > self.threshold,
                'reconstruction_error': error, 'threshold': self.threshold}

    def load_models(self,
                    if_model_path   ='models/iso_forest.pkl',
                    if_scaler_path  ='models/if_scaler.pkl',
                    lstm_model_path ='models/lstm_autoencoder.keras',
                    lstm_scaler_path='models/lstm_scaler.pkl',
                    threshold_path  ='models/lstm_threshold.pkl'):
        self.iso_forest  = joblib.load(if_model_path)
        self.if_scaler   = joblib.load(if_scaler_path)
        self._if_fitted  = True
        self.lstm_model  = load_model(lstm_model_path)
        self.lstm_scaler = joblib.load(lstm_scaler_path)
        self.threshold   = joblib.load(threshold_path)
        self._lstm_fitted = True
        print('All predictive maintenance models loaded successfully.')
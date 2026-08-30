"""
AQIEngine: Core business logic for the AQI Predictor.

This module encapsulates all Hopsworks connections, model loading,
and prediction logic. It can be used by:
  - FastAPI backend (backend/main.py) for API-based deployments
  - Streamlit frontend (frontend/app.py) for Streamlit Cloud deployments
"""

import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

LOCAL_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "feature_store", "karachi_daily_features.csv"
)
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Same display-name -> registry-name mapping used by Hopsworks mode, but
# pointing at the local .pkl files saved by ml_pipeline/train_model.py
LOCAL_MODEL_FILES = {
    "HGBR": "aqi_hgbr_models.pkl",
    "Random Forest": "aqi_rf_models.pkl",
    "Ridge Regression": "aqi_ridge_models.pkl",
    "Decision Tree": "aqi_dt_models.pkl",
    "Deep Learning (MLP)": "aqi_mlp_models.pkl",
    "HGBR (Optimized)": "aqi_hgbr_models.pkl",
    "Optimized": "aqi_hgbr_models.pkl",
}

# Also load Streamlit secrets if running on Streamlit Cloud.
# NOTE: `hasattr(st, "secrets")` is always True (st.secrets is a lazy
# object), so it does NOT guard against the "no secrets.toml found"
# case -- that only raises once you actually touch st.secrets. We
# guard with try/except instead, since secrets.toml legitimately
# doesn't exist in local/dev mode.
try:
    import streamlit as st
    for key in ["HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT"]:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            break  # no secrets.toml at all (e.g. local dev) -- fine, just skip
except ImportError:
    pass


def get_aqi_category(aqi):
    """Returns the human-readable AQI category."""
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


class AQIEngine:
    """
    Encapsulates all AQI prediction logic, including Hopsworks
    connectivity, model loading, and inference.
    """

    def __init__(self):
        self.models = {}
        self.feature_group = None
        self.use_hopsworks = False
        self._initialized = False

    def startup(self):
        """
        Initializes connections and loads available models.
        Uses Hopsworks (Feature Store + Model Registry) when
        HOPSWORKS_API_KEY is configured; otherwise falls back to the
        local CSV + locally-trained .pkl files under ./models/, so the
        dashboard is fully runnable before/without a Hopsworks account.
        """
        if self._initialized:
            return

        if os.getenv("HOPSWORKS_API_KEY"):
            try:
                self._startup_hopsworks()
                self._initialized = True
                return
            except Exception as e:
                print(f"Hopsworks startup failed ({e}); falling back to local mode.")

        self._startup_local()
        self._initialized = True

    def _startup_hopsworks(self):
        from data_pipeline.hopsworks_connector import get_feature_store
        from ml_pipeline.model_utils import load_latest_model_path

        print("Connecting to Hopsworks...")
        fs = get_feature_store()
        self.feature_group = fs.get_feature_group("karachi_aqi_daily", version=5)
        print("Feature group handle (v5) retrieved.")

        model_map = {
            "HGBR": "karachi_aqi_hgbr",
            "Random Forest": "karachi_aqi_rf",
            "Ridge Regression": "karachi_aqi_ridge",
            "Decision Tree": "karachi_aqi_dt",
            "Deep Learning (MLP)": "karachi_aqi_mlp",
            "HGBR (Optimized)": "karachi_aqi_hgbr",
            "Optimized": "karachi_aqi_hgbr",
        }

        print("Loading model architectures from Registry...")
        for display_name, registry_name in model_map.items():
            try:
                model_path = load_latest_model_path(registry_name)
                if model_path:
                    self.models[display_name] = joblib.load(model_path)
                    print(f"Loaded {display_name} ({registry_name})")
            except Exception as e:
                print(f"Warning: Could not load {display_name}: {e}")

        self.use_hopsworks = True

    def _startup_local(self):
        print("HOPSWORKS_API_KEY not set -- using local CSV + local models/ (dev mode).")
        self.use_hopsworks = False

        print("Loading model architectures from ./models/...")
        for display_name, filename in LOCAL_MODEL_FILES.items():
            path = os.path.join(LOCAL_MODELS_DIR, filename)
            if os.path.exists(path):
                self.models[display_name] = joblib.load(path)
                print(f"Loaded {display_name} ({filename})")
            else:
                print(f"Warning: {path} not found -- run "
                      f"`python -m ml_pipeline.train_model` first.")

    def _get_latest_data(self):
        """Fetches the latest rows, from Hopsworks Feature Group if
        connected, else from the local feature-store CSV."""
        if self.use_hopsworks and self.feature_group is not None:
            df = self.feature_group.read()
            df = df.sort_values("event_timestamp").reset_index(drop=True)
            return df

        if os.path.exists(LOCAL_CSV_PATH):
            df = pd.read_csv(LOCAL_CSV_PATH, parse_dates=["event_timestamp"])
            df.columns = df.columns.str.lower().str.replace(".", "_", regex=False)
            df = df.sort_values("event_timestamp").reset_index(drop=True)
            return df

        return None

    def get_current_aqi(self):
        """Returns the current AQI data dict."""
        df = self._get_latest_data()
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "timestamp": str(latest["event_timestamp"]),
                "aqi": round(float(latest["aqi"]), 2),
                "category": get_aqi_category(latest["aqi"]),
                "temperature": round(float(latest["temperature"]), 1),
                "humidity": round(float(latest["humidity"]), 1),
                "wind_speed": round(float(latest["windspeed"]), 1),
            }
        return {"error": "No data available from Feature Store"}

    def get_predictions(self, model_name="HGBR"):
        """Returns forecast predictions for the next 3 days."""
        df = self._get_latest_data()

        selected_model = self.models.get(model_name)
        if selected_model is None:
            selected_model = self.models.get("HGBR")

        if df is None or df.empty or selected_model is None:
            return {"error": f"Model '{model_name}' or features not available"}

        latest = df.iloc[-1]
        feature_cols = selected_model["features"]
        X_input = df[feature_cols].iloc[-1:]

        performance = selected_model.get("performance", {})

        predictions = []
        base_date = pd.to_datetime(latest["event_timestamp"])

        for i, horizon in enumerate(["1d", "2d", "3d"]):
            h_meta = selected_model["models"][horizon]
            m = h_meta["model"]
            s = h_meta["scaler"]

            X_scaled = s.transform(X_input)
            aqi_val = m.predict(X_scaled)[0]

            # Per-horizon metrics from training
            h_perf = performance.get(horizon, {})

            predictions.append({
                "date": (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "aqi": round(float(aqi_val), 2),
                "category": get_aqi_category(aqi_val),
                "r2": round(float(h_perf.get("R2", 0.0)), 3),
                "mae": round(float(h_perf.get("MAE", 0.0)), 2),
                "rmse": round(float(h_perf.get("RMSE", 0.0)), 2),
            })

        return {
            "model": model_name,
            "forecast": predictions,
        }

    def get_history(self, days=30):
        """Returns historical AQI data."""
        df = self._get_latest_data()
        if df is not None and not df.empty:
            history_df = df.tail(days)
            history = []
            for _, row in history_df.iterrows():
                history.append({
                    "date": str(row["event_timestamp"]),
                    "aqi": round(float(row["aqi"]), 2),
                })
            return history
        return []

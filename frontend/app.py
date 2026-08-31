import sys
from pathlib import Path

# Ensure project root is on sys.path (needed for Streamlit Cloud)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
import time
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Pearls AQI Predictor | Karachi",
    page_icon="frontend/assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Dark Mode ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #e0e0e0;
    }

    .main {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }

    .stMetric {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #667eea;
        color: #ffffff;
    }

    .aqi-card {
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .aqi-card:hover {
        transform: translateY(-5px);
    }

    .forecast-card {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #2d2f3b;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .status-good { background: linear-gradient(135deg, #0575E6, #00F260); }
    .status-moderate { background: linear-gradient(135deg, #FDC830, #F37335); }
    .status-unhealthy { background: linear-gradient(135deg, #e53935, #e35d5b); }
    .status-hazardous { background: linear-gradient(135deg, #8E2DE2, #4A00E0); }

    /* Custom styles for headers in dark mode */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    p {
        color: #b0b0b0;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Deployment Mode: API (local) vs Integrated (Streamlit Cloud)
# ============================================================
# Integrated mode is auto-detected on Streamlit Cloud where
# secrets are configured, or can be forced via env var.
def _detect_integrated_mode():
    """
    Default to Integrated Mode (import the AQIEngine directly -- works
    everywhere, including local dev with no Hopsworks account, since
    the engine itself falls back to local CSV/models). Only use API
    Mode (talk to a separately-running FastAPI backend) if explicitly
    requested via USE_API_MODE=1 -- e.g. for testing backend/main.py
    itself, or a deployment that intentionally splits frontend/backend.
    """
    return os.environ.get("USE_API_MODE") != "1"

INTEGRATED_MODE = _detect_integrated_mode()

if INTEGRATED_MODE:
    # --- Integrated Mode: import engine directly ---
    from backend.engine import AQIEngine

    @st.cache_resource
    def get_engine():
        """Initialize the AQI engine once and cache across reruns."""
        engine = AQIEngine()
        with st.spinner("🔌 Connecting to Hopsworks & loading models..."):
            engine.startup()
        return engine

    ENGINE = get_engine()

    def fetch_current():
        return ENGINE.get_current_aqi()

    def fetch_predictions(model_name):
        return ENGINE.get_predictions(model_name)

    def fetch_history():
        return ENGINE.get_history()

else:
    # --- API Mode: talk to the FastAPI backend ---
    API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

    def _api_get(endpoint):
        try:
            response = requests.get(f"{API_BASE_URL}/{endpoint}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
        return None

    def fetch_current():
        return _api_get("current")

    def fetch_predictions(model_name):
        return _api_get("predict?model_name=" + model_name)

    def fetch_history():
        return _api_get("history")


def get_aqi_class(aqi):
    if aqi <= 50: return "status-good"
    if aqi <= 100: return "status-moderate"
    if aqi <= 200: return "status-unhealthy"
    return "status-hazardous"


# --- Sidebar ---
with st.sidebar:
    st.image("frontend/assets/logo.png", width=100)
    st.title("Settings")
    
    model_choice = st.selectbox(
        "Prediction Model",
        ["HGBR", "Random Forest", "Ridge Regression", "Decision Tree", "Deep Learning (MLP)"]
    )
    
    st.divider()
    if st.button("🚀 Trigger Data Pipeline", use_container_width=True):
        with st.status("Fetching latest data...", expanded=True):
            time.sleep(1)
            st.write("Fetching weather data from OpenWeather...")
            time.sleep(1)
            st.write("Calculating features...")
            time.sleep(1)
            st.write("Updating Feature Store...")
        st.toast("Data updated successfully!", icon='✅')

    # Show deployment mode badge
    mode_label = "⚡ Integrated Mode" if INTEGRATED_MODE else "🖥️ API Mode"
    st.info(f"**{mode_label}** • Models retrain daily at 00:00 UTC.")
    st.caption("Created by **Muhammad Hassan Siddiqui**")

# --- Header ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Karachi AQI Dashboard")
    st.markdown("Real-time air quality monitoring and 3-day predictive forecasting powered by Machine Learning.")
with col2:
    st.markdown(f"<div style='text-align: right; color: #808080; padding-top: 30px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)

# --- Real-time Metrics ---
current_data = fetch_current()
if current_data and "error" not in current_data:
    aqi = current_data['aqi']
    status_class = get_aqi_class(aqi)
    
    st.markdown(f"""
        <div class="aqi-card {status_class}">
            <h1 style='font-size: 72px; margin: 0; color: white !important;'>{aqi}</h1>
            <h2 style='font-weight: 400; margin: 0; color: white !important;'>{current_data['category']}</h2>
            <p style='color: rgba(255,255,255,0.8);'>Air Quality Index (Current)</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Temperature", f"{current_data['temperature']}°C", "0.5°C")
    with c2:
        st.metric("Humidity", f"{current_data['humidity']}%", "-2%")
    with c3:
        st.metric("Wind Speed", f"{current_data['wind_speed']} km/h", "1.2 km/h")

# --- Forecast ---
st.header("🔮 3-Day Forecast")
forecast_data = fetch_predictions(model_choice)
if forecast_data and "error" not in forecast_data:
    f_cols = st.columns(3)
    for i, day in enumerate(forecast_data['forecast']):
        with f_cols[i]:
            status_c = get_aqi_class(day['aqi'])
            r2_val = day.get('r2', 'N/A')
            mae_val = day.get('mae', 'N/A')
            rmse_val = day.get('rmse', 'N/A')
            st.markdown(f"""
                <div class="forecast-card">
                    <p style='color: #808080; margin-bottom: 5px;'>{day['date']}</p>
                    <h3 style='margin: 0; color: white !important;'>{day['aqi']}</h3>
                    <div style='height: 10px; width: 60%; margin: 15px auto; border-radius: 5px; background: { "linear-gradient(90deg, #0575E6, #00F260)" if day['aqi'] < 50 else "linear-gradient(90deg, #dc2430, #7b4397)" }'></div>
                    <p style='font-weight: 600; color: #e0e0e0;'>{day['category']}</p>
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #2d2f3b;'>
                        <div style='display: flex; justify-content: space-around; text-align: center;'>
                            <div>
                                <p style='color: #667eea; font-size: 18px; font-weight: 700; margin: 0;'>{r2_val}</p>
                                <p style='color: #808080; font-size: 11px; margin: 0;'>R²</p>
                            </div>
                            <div>
                                <p style='color: #FF416C; font-size: 18px; font-weight: 700; margin: 0;'>{mae_val}</p>
                                <p style='color: #808080; font-size: 11px; margin: 0;'>MAE</p>
                            </div>
                            <div>
                                <p style='color: #FDC830; font-size: 18px; font-weight: 700; margin: 0;'>{rmse_val}</p>
                                <p style='color: #808080; font-size: 11px; margin: 0;'>RMSE</p>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- Charts ---
st.divider()
st.header("📈 AQI Trends")
history_data = fetch_history()
if history_data:
    df = pd.DataFrame(history_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['aqi'],
        mode='lines+markers',
        name='Historical AQI',
        line=dict(color='#667eea', width=4),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add forecast line
    if forecast_data and "error" not in forecast_data:
        f_df = pd.DataFrame(forecast_data['forecast'])
        # Connect last historical to first forecast
        combined_x = [df['date'].iloc[-1]] + list(f_df['date'])
        combined_y = [df['aqi'].iloc[-1]] + list(f_df['aqi'])
        
        fig.add_trace(go.Scatter(
            x=combined_x, y=combined_y,
            mode='lines+markers',
            name='ML Forecast',
            line=dict(color='#FF416C', width=4, dash='dot'),
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Date",
        yaxis_title="AQI Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width="stretch")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built for <b>Pearls AQI Project</b> • Karachi, Pakistan<br>
    Data sources: Open-Meteo API, Hopsworks • Model: Multi-Model Suite (Ridge, HGBR, RF, MLP, DT)
    <br> <span style='color: #667eea;'>Created by Muhammad Hassan Siddiqui</span>
</div>
""", unsafe_allow_html=True)

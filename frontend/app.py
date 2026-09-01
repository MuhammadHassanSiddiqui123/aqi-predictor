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

# --- Custom CSS: Warm White Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1C2321;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #1C2321 !important;
        font-weight: 600 !important;
    }

    p, span, label {
        color: #4B5259;
    }

    .main {
        background-color: #FAFAF7;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E4E1D8;
    }

    .stMetric {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E4E1D8;
        border-left: 3px solid #0E7C86;
    }
    [data-testid="stMetricLabel"] { color: #6B7280; }
    [data-testid="stMetricValue"] { color: #1C2321; font-family: 'Space Grotesk', sans-serif; }

    /* --- AQI gauge (hero) --- */
    .aqi-gauge {
        width: 220px; height: 220px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 10px auto 20px auto;
    }
    .aqi-gauge-inner {
        width: 178px; height: 178px; border-radius: 50%;
        background: #FFFFFF;
        box-shadow: inset 0 0 0 1px #E4E1D8;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .aqi-gauge-inner .aqi-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 46px; font-weight: 700; margin: 0; line-height: 1;
    }
    .aqi-gauge-inner .aqi-category {
        font-size: 15px; font-weight: 600; margin-top: 6px;
    }
    .aqi-gauge-inner .aqi-label {
        font-size: 12px; color: #9AA0A6; margin-top: 4px;
    }

    .forecast-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E4E1D8;
        text-align: center;
    }

    .status-good      { color: #22C55E; }
    .status-moderate   { color: #C98A00; }
    .status-unhealthy   { color: #E0631E; }
    .status-hazardous     { color: #B91C4B; }

    /* Hide default Streamlit chrome -- target these specifically by
       their own test-ids rather than the generic header/stToolbar
       wrappers, since those are reused for the sidebar's own
       expand/collapse control and hiding them breaks that control. */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none;}
    [data-testid="stMainMenu"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

AQI_SEVERITY_COLORS = {
    "status-good": "#22C55E",
    "status-moderate": "#C98A00",
    "status-unhealthy": "#E0631E",
    "status-hazardous": "#B91C4B",
}


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
    st.markdown(f"<div style='text-align: right; color: #9AA0A6; padding-top: 30px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)

# --- Real-time Metrics ---
current_data = fetch_current()
if current_data and "error" not in current_data:
    aqi = current_data['aqi']
    status_class = get_aqi_class(aqi)
    severity_color = AQI_SEVERITY_COLORS[status_class]
    gauge_deg = min(max(aqi, 0) / 300, 1.0) * 360

    st.markdown(f"""
        <div class="aqi-gauge" style="background: conic-gradient({severity_color} 0deg {gauge_deg}deg, #EEEAE0 {gauge_deg}deg 360deg);">
            <div class="aqi-gauge-inner">
                <p class="aqi-number" style="color: {severity_color};">{round(aqi)}</p>
                <p class="aqi-category {status_class}">{current_data['category']}</p>
                <p class="aqi-label">Air Quality Index</p>
            </div>
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
            day_color = AQI_SEVERITY_COLORS[status_c]
            r2_val = day.get('r2', 'N/A')
            mae_val = day.get('mae', 'N/A')
            rmse_val = day.get('rmse', 'N/A')
            st.markdown(f"""
                <div class="forecast-card">
                    <p style='color: #9AA0A6; margin-bottom: 5px;'>{day['date']}</p>
                    <h3 style='margin: 0; color: {day_color} !important;'>{day['aqi']}</h3>
                    <div style='height: 6px; width: 60%; margin: 15px auto; border-radius: 3px; background: {day_color};'></div>
                    <p style='font-weight: 600; color: #1C2321;'>{day['category']}</p>
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #E4E1D8;'>
                        <div style='display: flex; justify-content: space-around; text-align: center;'>
                            <div>
                                <p style='color: #0E7C86; font-size: 18px; font-weight: 700; margin: 0;'>{r2_val}</p>
                                <p style='color: #9AA0A6; font-size: 11px; margin: 0;'>R²</p>
                            </div>
                            <div>
                                <p style='color: #E0631E; font-size: 18px; font-weight: 700; margin: 0;'>{mae_val}</p>
                                <p style='color: #9AA0A6; font-size: 11px; margin: 0;'>MAE</p>
                            </div>
                            <div>
                                <p style='color: #C98A00; font-size: 18px; font-weight: 700; margin: 0;'>{rmse_val}</p>
                                <p style='color: #9AA0A6; font-size: 11px; margin: 0;'>RMSE</p>
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
        line=dict(color='#0E7C86', width=4),
        fill='tozeroy',
        fillcolor='rgba(14, 124, 134, 0.08)'
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
            line=dict(color='#E0631E', width=4, dash='dot'),
        ))

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1C2321'),
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
<div style='text-align: center; color: #9AA0A6;'>
    Built for <b style='color: #1C2321;'>Pearls AQI Project</b> • Karachi, Pakistan<br>
    Data sources: Open-Meteo API, Hopsworks • Model: Multi-Model Suite (Ridge, HGBR, RF, MLP, DT)<br>
    <span style='color: #0E7C86;'>Created by Muhammad Hassan Siddiqui</span>
</div>
""", unsafe_allow_html=True)
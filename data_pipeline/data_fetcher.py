import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# Karachi Coordinates
LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

def fetch_air_quality(start_date: str, end_date: str):
    """
    Fetch hourly air quality data from Open-Meteo.
    Dates format: YYYY-MM-DD
    """
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi"
        f"&timezone={TIMEZONE}"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        return None

def fetch_weather(start_date: str, end_date: str):
    """
    Fetch hourly weather data from Open-Meteo Archive.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        f"&timezone={TIMEZONE}"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_realtime_weather():
    """
    Fetch current weather (fallback to forecast API for 'current' since archive is for past)
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        f"&timezone={TIMEZONE}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("current")
    except Exception as e:
        print(f"Error fetching current weather: {e}")
        return None

if __name__ == "__main__":
    # Test fetch for last 3 days
    end = datetime.now().date()
    start = end - timedelta(days=3)
    
    print(f"Fetching data from {start} to {end}...")
    aq_df = fetch_air_quality(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    w_df = fetch_weather(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    
    if aq_df is not None and w_df is not None:
        merged = pd.merge(aq_df, w_df, on="time", how="inner")
        print("Success! Sample data:")
        print(merged.head())
    else:
        print("Fetch failed.")

import pandas as pd
import numpy as np

def aggregate_to_daily(df):
    """
    Process hourly data to daily aggregates.
    """
    if df is None or df.empty:
        return None
    
    # Aggregation rules
    agg_rules = {
        'us_aqi': 'mean',
        'pm2_5': 'mean',
        'pm10': 'mean',
        'nitrogen_dioxide': 'mean',
        'sulphur_dioxide': 'mean',
        'carbon_monoxide': 'mean',
        'ozone': 'mean',
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'precipitation': 'sum',
        'wind_speed_10m': 'mean'
    }
    
    # Resample to daily
    daily = df.set_index("time").resample("D").agg(agg_rules).round(3)
    
    # Rename for clarity
    daily = daily.rename(columns={
        'us_aqi': 'AQI',
        'pm2_5': 'PM2.5',
        'pm10': 'PM10',
        'nitrogen_dioxide': 'NO2',
        'sulphur_dioxide': 'SO2',
        'carbon_monoxide': 'CO',
        'ozone': 'O3',
        'temperature_2m': 'Temperature',
        'relative_humidity_2m': 'Humidity',
        'precipitation': 'Precipitation',
        'wind_speed_10m': 'WindSpeed'
    })
    
    daily["event_timestamp"] = daily.index
    return daily.reset_index(drop=True)

def add_temporal_features(df):
    """
    Create temporal features. Uses numeric labels instead of one-hot dummies 
    to keep the schema consistent across seasons.
    """
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    
    # 1. Standard Numeric Columns (1-column approach)
    df["month"] = df["event_timestamp"].dt.month
    df["weekday"] = df["event_timestamp"].dt.dayofweek + 1  # 1 (Mon) to 7 (Sun)
    
    # 2. Season Mapping (Ordinal: 1-Winter, 2-Spring, 3-Summer, 4-Autumn)
    season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4}
    df["season_idx"] = df["month"].map(season_map)
    
    # 3. Cyclical Encoding (Captures circular nature)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df["day_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    
    # 4. Log Transformations (Standard for skewed pollution data)
    df["log_PM2.5"] = np.log1p(df["PM2.5"])
    df["log_CO"] = np.log1p(df["CO"])
    
    # 5. Environmental Ratios
    # PM Ratio (PM2.5 / PM10) captures particle size distribution
    df["pm_ratio"] = df["PM2.5"] / df["PM10"].replace(0, np.nan)
    
    return df

def add_statistical_features(df):
    """
    Create lag and rolling features.
    """
    # 1. Advanced Lags (Yesterday's values for all pollutants)
    pollutants = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    for p in pollutants:
        df[f"{p}_lag_1"] = df[p].shift(1)
        if p == "AQI":
            df[f"{p}_lag_2"] = df[p].shift(2)
    
    # 2. Rolling averages (Short and Medium term trends)
    df["AQI_roll_mean_3"] = df["AQI"].rolling(window=3, min_periods=1).mean()
    df["AQI_roll_std_3"] = df["AQI"].rolling(window=3, min_periods=1).std()
    df["AQI_roll_mean_7"] = df["AQI"].rolling(window=7, min_periods=1).mean()
    
    # 3. Rate of Change (Daily differences)
    # Differences for pollutants
    for p in pollutants:
        df[f"{p}_diff"] = df[p].diff()
        
    # Differences for weather (captures sudden shifts like cold fronts or humidity spikes)
    df["temp_diff"] = df["Temperature"].diff()
    df["hum_diff"] = df["Humidity"].diff()
    
    return df

def add_target_features(df):
    """
    Create target variables for prediction (t+1, t+2, t+3).
    """
    df["target_aqi_1d"] = df["AQI"].shift(-1)
    df["target_aqi_2d"] = df["AQI"].shift(-2)
    df["target_aqi_3d"] = df["AQI"].shift(-3)
    return df

def clean_and_validate(df):
    """
    Remove outliers using IQR and handle missing values.
    """
    # IQR Capping for common pollutants and meteorological features
    cols_to_cap = ["PM2.5", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "Precipitation"]
    for col in cols_to_cap:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
            
    # Handle potential infinity/NaN from ratios or diffs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill to handle any gaps (crucial for lags)
    df = df.ffill().bfill()
    return df

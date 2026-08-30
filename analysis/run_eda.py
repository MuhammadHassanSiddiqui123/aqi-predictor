"""
run_eda.py  –  Exploratory Data Analysis for the Karachi AQI dataset.

Usage:
    python -m analysis.run_eda
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt

# ── Helpers ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feature_store", "karachi_daily_features.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "eda")

def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: CSV not found at {DATA_PATH}. Run the feature pipeline first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["event_timestamp"])
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")

    # ── 1.  Text summary ────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Data shape: {df.shape}\n")
        f.write(f"Date range: {df['event_timestamp'].min()} → {df['event_timestamp'].max()}\n\n")
        f.write("Columns:\n" + ", ".join(df.columns) + "\n\n")
        f.write("DTypes:\n" + str(df.dtypes) + "\n\n")
        f.write("Describe (numeric):\n" + str(df.describe()) + "\n\n")
        f.write("Missing values:\n" + str(df.isna().sum()) + "\n")
    print("  → summary.txt")

    # ── 2.  AQI time‑series ─────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(df["event_timestamp"], df["AQI"], linewidth=0.8, color="#667eea")
    plt.title("Daily AQI Over Time – Karachi")
    plt.xlabel("Date"); plt.ylabel("AQI")
    plt.xticks(rotation=45)
    _save("aqi_timeseries.png")
    print("  → aqi_timeseries.png")

    # ── 3.  Pollutant distributions (2×3 grid) ──────────────────────────────
    pollutants = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, pollutants):
        if col in df.columns:
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax, color="#667eea")
            ax.set_title(col)
    fig.suptitle("Pollutant Distributions", fontsize=14)
    _save("distributions.png")
    print("  → distributions.png")

    # ── 4.  AQI by month ────────────────────────────────────────────────────
    if "month" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="month", y="AQI", data=df, hue="month", palette="coolwarm", legend=False)
        plt.title("AQI by Month – Seasonal Patterns")
        plt.xlabel("Month"); plt.ylabel("AQI")
        _save("aqi_by_month.png")
        print("  → aqi_by_month.png")

    # ── 5.  AQI by weekday ──────────────────────────────────────────────────
    if "weekday" in df.columns:
        plt.figure(figsize=(10, 5))
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.boxplot(x="weekday", y="AQI", data=df, hue="weekday", palette="viridis", legend=False)
        plt.title("AQI by Day of Week")
        plt.xticks(range(7), day_names)
        plt.xlabel("Day of Week"); plt.ylabel("AQI")
        _save("aqi_by_weekday.png")
        print("  → aqi_by_weekday.png")

    # ── 6.  Correlation heatmap (key features vs targets) ───────────────────
    key_cols = [
        "AQI", "PM2.5", "PM10", "O3", "NO2", "SO2", "CO",
        "Temperature", "Humidity", "WindSpeed",
        "target_aqi_1d", "target_aqi_2d", "target_aqi_3d",
    ]
    available = [c for c in key_cols if c in df.columns]
    if available:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[available].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap – Key Features vs Targets")
        _save("correlation_heatmap.png")
        print("  → correlation_heatmap.png")

    # ── 7.  Rolling means + lag features over time ──────────────────────────
    roll_cols = [c for c in df.columns if "roll" in c.lower() and "aqi" in c.lower()]
    lag_cols  = [c for c in df.columns if "lag" in c.lower() and "aqi" in c.lower()]
    ts_cols = roll_cols + lag_cols
    if ts_cols:
        plt.figure(figsize=(12, 6))
        for col in ts_cols:
            plt.plot(df["event_timestamp"], df[col], label=col, linewidth=0.7)
        plt.title("AQI Rolling Means & Lag Features")
        plt.xlabel("Date"); plt.ylabel("AQI")
        plt.legend(fontsize=8)
        plt.xticks(rotation=45)
        _save("aqi_rolling_and_lags.png")
        print("  → aqi_rolling_and_lags.png")

    print(f"\n✅ EDA complete!  {len(os.listdir(OUTPUT_DIR))} files → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

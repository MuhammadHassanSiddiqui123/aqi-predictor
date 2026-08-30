import os
import pandas as pd
from datetime import datetime, timedelta
from data_pipeline.data_fetcher import fetch_air_quality, fetch_weather
from data_pipeline.engineer_features import (
    aggregate_to_daily, 
    add_temporal_features, 
    add_statistical_features, 
    add_target_features, 
    clean_and_validate
)

# Configuration
DATA_DIR = "data/feature_store"
FILE_PATH = os.path.join(DATA_DIR, "karachi_daily_features.csv")

# Raw/base columns (pre-feature-engineering). Only these are kept when
# merging with existing history -- derived columns (lags, rolling
# stats, targets, temporal encodings) are always recomputed fresh
# across the *full* merged series below, otherwise values near the
# old/new boundary would be wrong (e.g. a lag_1 computed only within
# the newly-fetched window, ignoring the day right before it).
_BASE_COLS = [
    "event_timestamp", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation", "WindSpeed",
]


def _load_existing_base_history() -> pd.DataFrame:
    """Load prior runs' data (if any) and strip back down to base
    columns, so it can be safely concatenated with a newly-fetched
    window before derived features are recomputed."""
    if not os.path.exists(FILE_PATH):
        return pd.DataFrame(columns=_BASE_COLS)
    existing = pd.read_csv(FILE_PATH)
    existing["event_timestamp"] = pd.to_datetime(existing["event_timestamp"])
    cols = [c for c in _BASE_COLS if c in existing.columns]
    return existing[cols]


def run_pipeline(days_back=30):
    """
    Orchestrates the data pipeline for a given historical range.

    IMPORTANT: this MERGES the newly-fetched window with any existing
    local history rather than overwriting it -- the GitHub Actions
    hourly job calls this with --days 1, which would otherwise wipe
    out the full historical CSV down to a single day on every run.
    """
    print(f"--- Starting AQI Pipeline (Last {days_back} days) ---")
    
    # 1. Fetch
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Step 1: Fetching raw data from {start_date} to {end_date}...")
    # Fetching in bulk is usually faster for Open-Meteo
    aq_df = fetch_air_quality(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    w_df = fetch_weather(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if aq_df is None or w_df is None:
        print("Pipeline aborted: Fetch failed.")
        return
    
    merged_hourly = pd.merge(aq_df, w_df, on="time", how="inner")
    
    # 2. Aggregate the newly-fetched window to daily base columns
    print("Step 2: Performing daily aggregation...")
    new_daily = aggregate_to_daily(merged_hourly)

    # 2b. Merge with existing local history (new data wins on overlap,
    # e.g. today's row being refreshed with more complete hourly data
    # later in the day), then recompute ALL derived features across
    # the full combined series so lags/rolling stats/targets are
    # correct at the old/new boundary.
    existing = _load_existing_base_history()
    combined_base = pd.concat([existing, new_daily[_BASE_COLS]], ignore_index=True)
    combined_base["event_timestamp"] = pd.to_datetime(combined_base["event_timestamp"])
    combined_base = (
        combined_base.drop_duplicates(subset="event_timestamp", keep="last")
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )
    print(f"  Merged: {len(existing)} existing + {len(new_daily)} fetched "
          f"-> {len(combined_base)} total rows after de-dup.")

    print("Step 3: Recomputing engineered features across full history...")
    df = add_temporal_features(combined_base)
    df = add_statistical_features(df)
    df = add_target_features(df)
    df = clean_and_validate(df)
    
    # 4. Save to Local Feature Store (CSV) -- now a legitimate full
    # overwrite, since `df` already contains the merged history.
    print(f"Step 4: Saving {len(df)} records to {FILE_PATH}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FILE_PATH, index=False)
    
    # 5. Upload to Hopsworks Cloud (insert() upserts by primary key, so
    # sending the full history is safe and idempotent)
    print("Step 5: Uploading to Hopsworks Cloud...")
    try:
        from data_pipeline.hopsworks_connector import create_or_get_feature_group
        # Ensure UTC and correct metadata for Hopsworks
        upload_df = df.copy()
        upload_df["event_timestamp"] = pd.to_datetime(upload_df["event_timestamp"])
        upload_df["karachi_id"] = "karachi_001"
        create_or_get_feature_group(upload_df)
        print("Success: Data synced with Hopsworks.")
    except Exception as e:
        print(f"Warning: Hopsworks upload failed: {e}")

    print("--- Pipeline Completed Successfully ---")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AQI Data Pipeline")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch back")
    args = parser.parse_args()
    
    run_pipeline(days_back=args.days)

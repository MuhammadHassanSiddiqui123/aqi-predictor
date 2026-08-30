import hopsworks
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_feature_store():
    """
    Connects to Hopsworks and returns the feature store handle.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "Pearls_AQI_Predictor")
    
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables.")
    
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    return project.get_feature_store()

def create_or_get_feature_group(df, name="karachi_aqi_daily", version=5): 
    """
    Creates or retrieves a feature group in Hopsworks.
    """
    # resolves hopsworks requirement of not having a '.' in column names
    df.columns = df.columns.str.lower().str.replace('.', '_', regex=False)
    
    # Create a date string column (YYYY-MM-DD) from event_timestamp for primary key
    df['date'] = pd.to_datetime(df['event_timestamp']).dt.date.astype(str)
    
    fs = get_feature_store()
    
    # Primary key: location + date string
    primary_key = ["karachi_id", "date"]
    
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        primary_key=primary_key,
        description="Daily Karachi AQI and Weather features with derived temporal and lag statistics.",
        online_enabled=True,
        event_time="event_timestamp",
        time_travel_format="HUDI"
    )
    
    fg.insert(df)
    return fg

if __name__ == "__main__":
    import pandas as pd
    # Test connection and local file upload
    FILE_PATH = "data/feature_store/karachi_daily_features.csv"
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        

        # Ensure timestamp is datetime and UTC (Hopsworks requirement)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.tz_localize(None) 
        df["karachi_id"] = "karachi_001" # Ensure metadata exists
        
        print(f"Uploading {len(df)} records to Hopsworks...")
        fg = create_or_get_feature_group(df)
        print("Upload successful!")
    else:
        print("Local feature file not found. Run pipeline_manager first.")

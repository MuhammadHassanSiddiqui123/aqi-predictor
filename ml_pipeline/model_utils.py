import os
import joblib
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def get_model_registry():
    """
    Connects to Hopsworks and returns the model registry handle.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "Pearls_AQI_Predictor")
    
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    return project.get_model_registry()

def save_model_to_registry(model_path, model_name, metrics, description):
    """
    Uploads a model file to the Hopsworks Model Registry.
    """
    mr = get_model_registry()
    
    # Create the model metadata
    aqi_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description
    )
    
    # Export and save
    aqi_model.save(model_path)
    print(f"Model '{model_name}' successfully saved to registry!")
    return aqi_model

def load_latest_model_path(model_name):
    """
    Downloads the highest version of a model from the registry.
    """
    mr = get_model_registry()
    
    # Get all models with this name and find the max version
    models = mr.get_models(model_name)
    if not models:
        print(f"No models found for name: {model_name}")
        return None
        
    # Find the model with the highest version
    latest_model = max(models, key=lambda m: m.version)
    print(f"Fetching latest model: {model_name} (Version {latest_model.version})")
    
    model_dir = latest_model.download()
    
    # Return the path to the .pkl file inside the downloaded directory
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            path = os.path.join(model_dir, file)
            print(f"Model file found: {file}")
            return path
    
    return None

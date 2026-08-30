from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.engine import AQIEngine

app = FastAPI(title="Pearls AQI Predictor API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared engine instance
engine = AQIEngine()

@app.on_event("startup")
async def startup_event():
    """Initializes Hopsworks connections and loads available models."""
    engine.startup()

@app.get("/")
async def root():
    return {"message": "Welcome to Pearls AQI Predictor API", "city": "Karachi", "status": "Multi-Model Support Active"}

@app.get("/current")
async def get_current_aqi():
    return engine.get_current_aqi()

@app.get("/predict")
async def get_predictions(model_name: str = "HGBR"):
    return engine.get_predictions(model_name)

@app.get("/history")
async def get_history(days: int = 30):
    return engine.get_history(days)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

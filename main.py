from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging
import threading
from datetime import datetime, timedelta

app = FastAPI(title="COVID-19 Forecast API")

# Lazy-loaded resources + thread-safety
logger = logging.getLogger("uvicorn.error")
_load_lock = threading.Lock()
models = None
model_load_error = None
df = None
df_load_error = None

def ensure_models_loaded():
    """Load models into the global `models` variable if not already loaded."""
    global models, model_load_error
    if models is not None:
        return
    with _load_lock:
        if models is not None:
            return
        try:
            loaded = joblib.load("covid_forecast_models.pkl")
            models = loaded
            model_load_error = None
            logger.info("Models loaded successfully")
        except Exception as e:
            models = {}
            model_load_error = str(e)
            logger.warning("Could not load model file: %s", model_load_error)

def ensure_data_loaded():
    """Load dataset into the global `df` variable if not already loaded."""
    global df, df_load_error
    if df is not None:
        return
    with _load_lock:
        if df is not None:
            return
        try:
            loaded_df = pd.read_csv("cleaned_covid_data.csv")
            loaded_df['date'] = pd.to_datetime(loaded_df['date'])
            df = loaded_df
            df_load_error = None
            logger.info("Data loaded successfully")
        except Exception as e:
            df = None
            df_load_error = str(e)
            logger.warning("Could not load data file: %s", df_load_error)

@app.post("/predict")
def predict(country: str, days: int = 7):
    # Ensure resources are loaded lazily on first request
    ensure_models_loaded()
    ensure_data_loaded()

    if not models:
        # Models not loaded (missing file or load error)
        raise HTTPException(status_code=503, detail=f"Model not available on server: {model_load_error}")

    if df is None:
        # Data file not loaded
        raise HTTPException(status_code=503, detail=f"Training data not available on server: {df_load_error}")

    if country not in models:
        raise HTTPException(status_code=404, detail="Country not found or insufficient data")
    
    if days < 1 or days > 14:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 14")
    
    # Get last 7 days of actual data for this country
    country_data = df[df['location'] == country].sort_values('date')
    last_values = country_data['new_cases'].tail(7).values
    
    if len(last_values) < 7:
        raise HTTPException(status_code=500, detail="Insufficient historical data")

    forecast = []
    current_input = last_values.copy()

    model = models[country]
    for _ in range(days):
        pred = model.predict([current_input])[0]
        pred = max(0, int(pred))  # no negative cases
        forecast.append(pred)
        # shift window
        current_input = list(current_input[1:]) + [pred]
    
    return {"country": country, "forecast_days": days, "forecast": forecast}
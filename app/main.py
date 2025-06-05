
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import joblib
import os

# Constants
CSV_PATH = "./data/full_grouped(1).csv"
MODEL_PATH = "xgb_model.json"
base_date = datetime(2020, 1, 1)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Functions
def convert_id_calendar_to_date(id_calendar_value):
    try:
        days_since_base = int(id_calendar_value) - 1
        return base_date + timedelta(days=days_since_base)
    except (ValueError, TypeError):
        return pd.NaT

def rmse_percent(y_true, y_pred):
    return 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)

@app.get("/predict")
def predict(country: int):
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture CSV : {e}")

    if 'id_calendar' not in df.columns:
        raise HTTPException(status_code=500, detail="Colonne 'id_calendar' manquante.")

    df['date_value'] = df['id_calendar'].apply(convert_id_calendar_to_date)
    if df['date_value'].isna().all():
        raise HTTPException(status_code=500, detail="Toutes les dates sont NaT.")

    df_country = df[df['id'] == country].copy()
    if df_country.empty:
        raise HTTPException(status_code=404, detail=f"Aucune donnée pour le pays ID {country}.")

    df_country = df_country.dropna(subset=['date_value'])
    df_country.sort_values('date_value', inplace=True)

    df_country['dayofyear'] = df_country['date_value'].dt.dayofyear
    df_country['dayofweek'] = df_country['date_value'].dt.dayofweek
    df_country['month'] = df_country['date_value'].dt.month
    df_country['year'] = df_country['date_value'].dt.year
    df_country['weekofyear'] = df_country['date_value'].dt.isocalendar().week.astype(int)
    df_country['quarter'] = df_country['date_value'].dt.quarter

    for lag in range(1, 8):
        df_country[f'new_cases_lag{lag}'] = df_country['new_cases'].shift(lag)
    df_country.dropna(inplace=True)

    features = ['dayofyear', 'dayofweek', 'month', 'year', 'weekofyear', 'quarter'] +                [f'new_cases_lag{lag}' for lag in range(1, 8)]

    if not os.path.exists(MODEL_PATH):
        X = df_country[features]
        y = df_country['new_cases']
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        model.save_model(MODEL_PATH)
    else:
        model = XGBRegressor()
        model.load_model(MODEL_PATH)

    X_pred = df_country[features]
    y_pred = model.predict(X_pred)

    return {
        "country": int(country),
        "predictions": y_pred[-10:].tolist()
    }

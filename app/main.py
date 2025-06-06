from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

CSV_PATH = "./data/full_grouped(1).csv"
base_date = datetime(2020, 1, 1)
future_days = 30

def convert_id_calendar_to_date(id_calendar_value):
    try:
        days_since_base = int(id_calendar_value) - 1
        return base_date + timedelta(days=days_since_base)
    except (ValueError, TypeError):
        return pd.NaT

def rmse_percent(y_true, y_pred):
    return 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/predict")
def predict(country: int):
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture CSV : {e}")

    if 'id_calendar' not in df.columns:
        raise HTTPException(status_code=500, detail="Colonne 'id_calendar' manquante.")

    df['date_value'] = df['id_calendar'].apply(convert_id_calendar_to_date)
    df_country = df[df['id'] == country].copy()
    df_country = df_country.dropna(subset=['date_value'])
    df_country.sort_values('date_value', inplace=True)
    if df_country.empty:
        raise HTTPException(status_code=500, detail="Données invalides après nettoyage.")

    country_name = df_country['country'].iloc[0]

    df_country['dayofyear'] = df_country['date_value'].dt.dayofyear
    df_country['dayofweek'] = df_country['date_value'].dt.dayofweek
    df_country['month'] = df_country['date_value'].dt.month
    df_country['year'] = df_country['date_value'].dt.year
    df_country['weekofyear'] = df_country['date_value'].dt.isocalendar().week.astype(int)
    df_country['quarter'] = df_country['date_value'].dt.quarter

    for lag in range(1, 8):
        df_country[f'new_cases_lag{lag}'] = df_country['new_cases'].shift(lag).fillna(0)
        df_country[f'new_deaths_lag{lag}'] = df_country['new_deaths'].shift(lag).fillna(0)

    df_country['new_cases'] = pd.to_numeric(df_country['new_cases'], errors='coerce')
    df_country['new_deaths'] = pd.to_numeric(df_country['new_deaths'], errors='coerce')
    df_country['total_cases'] = pd.to_numeric(df_country['total_cases'], errors='coerce')
    df_country['total_deaths'] = pd.to_numeric(df_country['total_deaths'], errors='coerce')

    feature_base = ['dayofyear', 'dayofweek', 'month', 'year', 'id_pandemic', 'weekofyear', 'quarter']
    lag_features = [f'new_cases_lag{l}' for l in range(1, 8)] + [f'new_deaths_lag{l}' for l in range(1, 8)]
    features_cols = feature_base + lag_features

    df_country = df_country.dropna(subset=features_cols + ['new_cases', 'new_deaths', 'total_cases', 'total_deaths'])
    df_country = df_country[
        ~df_country[['new_cases', 'new_deaths', 'total_cases', 'total_deaths']].isin([np.inf, -np.inf]).any(axis=1)
    ]

    if df_country.empty:
        raise HTTPException(status_code=500, detail="Pas assez de données valides après nettoyage.")

    X = df_country[features_cols]
    y_new_cases = np.log1p(df_country['new_cases'].ewm(span=3).mean())
    y_new_deaths = np.log1p(df_country['new_deaths'].ewm(span=3).mean())
    y_total_cases = np.log1p(df_country['total_cases'])
    y_total_deaths = np.log1p(df_country['total_deaths'])

    try:
        X_train, X_test, y_train_nc, y_test_nc = train_test_split(X, y_new_cases, test_size=0.2, random_state=42)
        _, _, y_train_nd, y_test_nd = train_test_split(X, y_new_deaths, test_size=0.2, random_state=42)
        _, _, y_train_tc, y_test_tc = train_test_split(X, y_total_cases, test_size=0.2, random_state=42)
        _, _, y_train_td, y_test_td = train_test_split(X, y_total_deaths, test_size=0.2, random_state=42)

        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42
        }

        model_new_cases = XGBRegressor(**params)
        model_new_cases.fit(X_train, y_train_nc)

        model_new_deaths = XGBRegressor(**params)
        model_new_deaths.fit(X_train, y_train_nd)

        model_total_cases = XGBRegressor(**params)
        model_total_cases.fit(X_train, y_train_tc)

        model_total_deaths = XGBRegressor(**params)
        model_total_deaths.fit(X_train, y_train_td)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur modèle : {e}")

    last_row = df_country.iloc[-1]
    last_date = df_country['date_value'].max()
    last_total_cases = last_row['total_cases']
    last_total_deaths = last_row['total_deaths']
    last_new_cases = last_row['new_cases']
    last_new_deaths = last_row['new_deaths']
    last_id_pandemic = last_row['id_pandemic']

    predictions = []
    for i in range(1, future_days + 1):
        current_date = last_date + timedelta(days=i)
        features = pd.DataFrame([{
            'dayofyear': current_date.timetuple().tm_yday,
            'dayofweek': current_date.weekday(),
            'month': current_date.month,
            'year': current_date.year,
            'id_pandemic': int(last_id_pandemic),
            'weekofyear': current_date.isocalendar().week,
            'quarter': (current_date.month - 1) // 3 + 1,
            **{f'new_cases_lag{l}': last_new_cases for l in range(1, 8)},
            **{f'new_deaths_lag{l}': last_new_deaths for l in range(1, 8)}
        }])

        pred_new_cases = max(0, float(np.expm1(model_new_cases.predict(features)[0])))
        pred_new_deaths = max(0, float(np.expm1(model_new_deaths.predict(features)[0])))
        pred_total_cases = last_total_cases + pred_new_cases
        pred_total_deaths = last_total_deaths + pred_new_deaths

        predictions.append({
            "date": current_date.strftime('%Y-%m-%d'),
            "new_cases": round(pred_new_cases, 2),
            "new_deaths": round(pred_new_deaths, 2),
            "total_cases": round(pred_total_cases),
            "total_deaths": round(pred_total_deaths)
        })

        last_new_cases = pred_new_cases
        last_new_deaths = pred_new_deaths
        last_total_cases = pred_total_cases
        last_total_deaths = pred_total_deaths

    return {
        "country": country_name,
        "country_id": country,
        "predictions": predictions
    }

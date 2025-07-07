import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.utils import engineer_features

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def forecast_with_model(df, freq_choice, model_type, forecast_horizon):
    future_dates = pd.date_range(start=df['date'].iloc[-1], periods=forecast_horizon + 1, freq=_freq_map(freq_choice))[1:]

    if model_type == "ARIMA":
        model = auto_arima(df['sales'], seasonal=False, suppress_warnings=True)
        forecast = model.predict(n_periods=forecast_horizon)
        return pd.DataFrame({'date': future_dates, 'forecasted_sales': forecast}), evaluate_model(df['sales'][-forecast_horizon:], forecast)

    elif model_type == "Prophet":
        df_prophet = df.rename(columns={"date": "ds", "sales": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_horizon, freq=_freq_map(freq_choice))
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecasted_sales'})
        return forecast_df.tail(forecast_horizon), evaluate_model(df['sales'][-forecast_horizon:], forecast_df['forecasted_sales'].tail(forecast_horizon).values)

    else:
        df_feat = engineer_features(df)
        df_feat.dropna(inplace=True)

        if len(df_feat) < forecast_horizon + 1:
            raise ValueError("Not enough data after feature engineering to train the model.")

        X = df_feat.drop(columns=['sales', 'date'])
        y = df_feat['sales']
        X_train = X[:-forecast_horizon]
        y_train = y[:-forecast_horizon]

        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
        elif model_type == "XGBoost":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1)

        model.fit(X_train, y_train)

        last_known = X.iloc[-1].values.tolist()
        forecast = []
        for _ in range(forecast_horizon):
            pred = model.predict([last_known])[0]
            forecast.append(pred)
            last_known = last_known[1:] + [pred]

        return pd.DataFrame({'date': future_dates, 'forecasted_sales': forecast}), evaluate_model(y[-forecast_horizon:], forecast)

def _freq_map(freq):
    return {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Annually": "A"}[freq]

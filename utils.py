import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(df, freq_choice):
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Annually": "A"}
    df = df.set_index('date').resample(freq_map[freq_choice]).sum().reset_index()
    return df

def create_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, lags=7):
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['sales'].shift(i)
    return df

def create_rolling_features(df, windows=[3, 7]):
    for window in windows:
        df[f'roll_mean_{window}'] = df['sales'].shift(1).rolling(window).mean()
        df[f'roll_std_{window}'] = df['sales'].shift(1).rolling(window).std()
    return df

def engineer_features(df, lags=7):
    df = create_time_features(df)
    df = create_lag_features(df, lags)
    df = create_rolling_features(df)
    # Include any custom variables already in the dataset
    return df

def plot_forecast(df, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'], df['sales'], label="Historical Sales")
    ax.plot(forecast_df['date'], forecast_df['forecasted_sales'], label="Forecast", linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    return fig

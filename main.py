import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app.model import forecast_with_model
from app.utils import preprocess_data, plot_forecast

st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("📈 Advanced Forecasting App")

uploaded_file = st.file_uploader("Upload a CSV with 'date', 'sales', and optional custom variables", type=["csv"])
freq_choice = st.selectbox("Select Forecast Frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"])
model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "Random Forest", "XGBoost", "Prophet"])
forecast_horizon = st.slider("Select Forecast Horizon", min_value=5, max_value=60, value=10)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_prepped = preprocess_data(df, freq_choice)

    st.subheader("📊 Aggregated Data")
    st.dataframe(df_prepped.tail())

    try:
        forecast_df, metrics = forecast_with_model(df_prepped, freq_choice, model_choice, forecast_horizon)

        st.subheader(f"📉 Forecast Plot ({model_choice})")
        st.caption(f"Forecasting {forecast_horizon} future {freq_choice.lower()} periods from {df_prepped['date'].iloc[-1].date()}")
        fig = plot_forecast(df_prepped, forecast_df)
        st.pyplot(fig)

        st.subheader("📅 Forecasted Values")
        st.dataframe(forecast_df)

        st.subheader("📈 Forecast Accuracy")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col2.metric("MAE", f"{metrics['MAE']:.2f}")
        col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        st.subheader("📥 Download Forecast")
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast as CSV", data=csv, file_name='forecasted_sales.csv', mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error: {e}")



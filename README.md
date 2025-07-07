# 📈 Forecasting App

A simple Streamlit app for forecasting sales using XGBoost or Random Forest as a fallback.

## Features
- Upload CSV with `date` and `sales`
- Choose frequency: daily, weekly, monthly, quarterly, annually
- Forecast future values for the same time period
- Visualize and download forecasted results

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from datetime import date

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Weather Prediction", layout="wide")

# -------------------------------
# Load model & local dataset
# -------------------------------
@st.cache_resource
def load_model():
    model = load("weather_prediction_model.joblib")
    return model


@st.cache_data
def load_climate_data():
    # Load CSV and parse the date index
    df = pd.read_csv("local_weather.csv", index_col='DATE', parse_dates=True)

    # Clean and rename columns
    df = df[['PRCP', 'TMAX', 'TMIN']].copy()
    df.columns = ['precip', 'temp_max', 'temp_min']
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Extract time features
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear

    # Compute seasonal averages
    df['monthly_avg'] = df['temp_max'].groupby(df['month']).transform('mean')
    df['day_of_year_avg'] = df['temp_max'].groupby(df['day_of_year']).transform('mean')

    return df


# Load resources
model = load_model()
climate_data = load_climate_data()

st.sidebar.success("✅ Model & dataset loaded successfully!")

# -------------------------------
# App title and intro
# -------------------------------
st.title("🌤️ Weather Prediction using Ridge Regression")
st.write("""
This app predicts **tomorrow's maximum temperature** using a trained Ridge Regression model.  
It automatically performs the same feature engineering used during model training — including monthly and seasonal averages from a local dataset.
""")

# -------------------------------
# User input section
# -------------------------------
st.sidebar.header("Input Today's Weather Data")

precip = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
temp_max = st.sidebar.number_input("Today's Max Temperature (°C)", min_value=-20.0, max_value=60.0, value=30.0, step=0.5)
temp_min = st.sidebar.number_input("Today's Min Temperature (°C)", min_value=-30.0, max_value=40.0, value=20.0, step=0.5)
yesterday_temp_max = st.sidebar.number_input("Yesterday's Max Temperature (°C)", min_value=-20.0, max_value=60.0, value=30.0, step=0.5)

# -------------------------------
# Create input DataFrame
# -------------------------------
input_df = pd.DataFrame({
    'precip': [precip],
    'temp_max': [temp_max],
    'temp_min': [temp_min],
    'yesterday_temp_max': [yesterday_temp_max]
})

st.subheader("Raw Input Summary")
st.dataframe(input_df)

# -------------------------------
# Feature engineering (with local dataset)
# -------------------------------
def engineer_features(df):
    df = df.copy()

    # Base engineered features
    df['month_max'] = df['temp_max']
    df['month_day_max'] = df['month_max'] / df['temp_max']
    df['max_min'] = df['temp_max'] / df['temp_min']
    df['temp_max_ramp'] = df['temp_max'] - df['yesterday_temp_max']

    # Use the latest date from your dataset to ensure coverage
    latest_date = climate_data.index.max()
    month = latest_date.month
    day_of_year = latest_date.dayofyear

    # Retrieve precomputed averages

    month_data = climate_data.loc[climate_data['month'] == month, 'monthly_avg']
    day_data = climate_data.loc[climate_data['day_of_year'] == day_of_year, 'day_of_year_avg']

    # Use first available value if exists, otherwise fallback to global mean
    monthly_avg = month_data.iloc[0] if not month_data.empty else climate_data['monthly_avg'].mean()
    day_of_year_avg = day_data.iloc[0] if not day_data.empty else climate_data['day_of_year_avg'].mean()

    df['monthly_avg'] = monthly_avg
    df['day_of_year_avg'] = day_of_year_avg
    # Drop columns not seen by model
    df = df.drop(columns=['yesterday_temp_max'], errors='ignore')

    return df

features = engineer_features(input_df)

st.subheader("Engineered Features (used by the model)")
st.dataframe(features)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌈 Predict Tomorrow's Max Temperature"):
    # ✅ Align features to match training order
    expected_cols = list(model.feature_names_in_)
    features = features[expected_cols]

    # ✅ Sanitize NaNs just in case
    if features.isna().any().any():
        st.warning("⚠️ Missing values detected — replacing with column means for safety.")
        features = features.fillna(features.mean())

    # Predict
    prediction = model.predict(features)[0]

    st.success(f"**Predicted Max Temperature for Tomorrow: {prediction:.2f}°C**")
    st.caption("Model: Ridge Regression trained with historical weather features")

    # Debug panel (optional)
    st.markdown("### 🔍 Debug Info")
    st.write("**Feature Order Check:**")
    st.code(expected_cols)
    st.write("**Current Features:**")
    st.dataframe(features)

# -------------------------------
# Notes
# -------------------------------
st.markdown("---")
st.markdown("""
### 📝 Notes:
This app automatically performs the same feature engineering as used during model training:
- `month_day_max = month_max / temp_max`
- `max_min = temp_max / temp_min`
- `temp_max_ramp = temp_max - yesterday_temp_max`
- `monthly_avg` and `day_of_year_avg` are retrieved from your **local dataset (`local_weather.csv`)**
""")


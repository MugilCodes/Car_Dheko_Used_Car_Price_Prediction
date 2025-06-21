import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load cleaned data
df = pd.read_csv("cleaned_car_data.csv")

# Load model and expected columns
with open('XGBoost_best_car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_features_columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)

st.set_page_config(layout="wide")
st.title("🚗 Used Car Price Prediction - Car Dheko")

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", sorted(df['brand'].dropna().unique()))
        model_name = st.selectbox("Model", sorted(df[df['brand'] == brand]['model'].dropna().unique()))
        variant = st.selectbox("Variant", sorted(df[df['model'] == model_name]['variant'].dropna().unique()))
        fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_type'].dropna().unique()))
        body_type = st.selectbox("Body Type", sorted(df['body_type'].dropna().astype(str).unique()))
        transmission = st.selectbox("Transmission", sorted(df['transmission'].dropna().unique()))
        gear_box = st.selectbox("Gear Box", sorted(df['gear_box'].dropna().unique()))
        drive_type = st.selectbox("Drive Type", sorted(df['drive_type'].dropna().unique()))
        engine_type = st.selectbox("Engine Type", sorted(df['engine_type'].dropna().unique()))
        steering_type = st.selectbox("Steering Type", sorted(df['steering_type'].dropna().unique()))
        owner_number = st.slider("Owner Number", 1, 5, 1)
        kms_driven = st.number_input("Kilometers Driven", 100, 500000, 30000)

    with col2:
        model_year = st.slider("Model Year", 2000, 2025, 2018)
        registration_year = st.slider("Registration Year", 2000, 2025, model_year)
        engine_displacement = st.number_input("Engine Displacement (cc)", 500, 5000, 1200)
        mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 18.0)
        max_power = st.number_input("Max Power (bhp)", 20.0, 500.0, 85.0)
        torque = st.number_input("Torque (Nm)", 20.0, 800.0, 100.0)
        top_speed = st.number_input("Top Speed (km/h)", 80.0, 300.0, 160.0)
        acceleration = st.number_input("0–100 km/h Acceleration (sec)", 5.0, 20.0, 12.0)
        color = st.selectbox("Color", sorted(df['color'].dropna().unique()))
        city = st.selectbox("City", sorted(df['City'].dropna().unique()))

    car_age = 2025 - model_year
    submitted = st.form_submit_button("🔍 Predict Price")

if submitted:
    input_dict = {
        'owner_number': owner_number,
        'kms_driven': kms_driven,
        'model_year': model_year,
        'registration_year': registration_year,
        'engine_displacement': engine_displacement,
        'mileage': mileage,
        'max_power': max_power,
        'torque': torque,
        'top_speed': top_speed,
        'acceleration': acceleration,
        'car_age': car_age,
        f'brand_{brand}': 1,
        f'model_{model_name}': 1,
        f'variant_{variant}': 1,
        f'fuel_type_{fuel_type}': 1,
        f'body_type_{body_type}': 1,
        f'transmission_{transmission}': 1,
        f'gear_box_{gear_box}': 1,
        f'drive_type_{drive_type}': 1,
        f'engine_type_{engine_type}': 1,
        f'steering_type_{steering_type}': 1,
        f'color_{color}': 1,
        f'City_{city}': 1
    }

    input_df = pd.DataFrame([np.zeros(len(expected_columns))], columns=expected_columns)

    for key, val in input_dict.items():
        if key in input_df.columns:
            input_df.at[0, key] = val

    with st.expander("🔎 Selected Inputs Summary"):
        st.json(input_dict)

    try:
        price = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: ₹ {round(price):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# LOAD MODEL ARTIFACTS
# ----------------------------
try:
    model = joblib.load("model.pkl.gz")
    features = joblib.load("features.pkl")
    threshold = joblib.load("threshold.pkl")
except Exception as e:
    st.error("Model files not found. Please check deployment.")
    st.stop()

st.title("Credit Card Fraud Detection System")

st.write("Enter basic transaction details")

# -------- USER INPUTS --------

amt = st.number_input("Transaction Amount", min_value=0.0)
age = st.number_input("Customer Age", min_value=18, max_value=100)
distance = st.number_input("Distance to Merchant (km)", min_value=0.0)

hour = st.slider("Transaction Hour", 0, 23)
day = st.slider("Day of Month", 1, 31)
month = st.slider("Month", 1, 12)

category = st.selectbox("Transaction Category", [
    "food_dining","gas_transport","grocery_net","grocery_pos",
    "health_fitness","home","kids_pets","misc_net",
    "misc_pos","personal_care","shopping_net",
    "shopping_pos","travel"
])

gender = st.selectbox("Gender", ["M","F"])

# -------- BUILD INPUT DATAFRAME --------

input_dict = {feature: 0 for feature in features}

input_dict["amt"] = amt
input_dict["age"] = age
input_dict["distance"] = distance
input_dict["hour"] = hour
input_dict["day"] = day
input_dict["month"] = month

# Handle category dummy
cat_col = f"category_{category}"
if cat_col in input_dict:
    input_dict[cat_col] = 1

# Handle gender dummy
if gender == "M" and "gender_M" in input_dict:
    input_dict["gender_M"] = 1

input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df[features]

# -------- PREDICTION --------


if st.button("Predict Fraud"):

    try:
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability > threshold else 0

        st.write("Fraud Probability:", round(probability, 4))

        if prediction == 1:
            st.error("Fraudulent Transaction Detected")
        else:
            st.success("Legitimate Transaction")

    except Exception as e:
        st.error("Prediction failed. Check feature alignment.")

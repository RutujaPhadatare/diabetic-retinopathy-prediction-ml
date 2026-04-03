import streamlit as st
import joblib
import numpy as np

model = joblib.load("retinopathy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetic Retinopathy Prediction")

st.write("Enter patient details")

age = st.number_input("Age", min_value=1.0, max_value=120.0, value=60.0)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=50.0, max_value=200.0, value=100.0)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40.0, max_value=150.0, value=90.0)
cholesterol = st.number_input("Cholesterol", min_value=50.0, max_value=300.0, value=100.0)

if st.button("Predict"):

    input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    if prediction[0] == 1:
        st.error(f"Patient likely has Diabetic Retinopathy (Probability: {probability[0][1]:.2f})")
    else:
        st.success(f"Patient likely does NOT have Diabetic Retinopathy (Probability: {probability[0][0]:.2f})")

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Medi-Predict", page_icon="🩺", layout="centered")

st.title("🩺 Medi-Predict - Diabetes Risk Detector")

# Sidebar inputs
st.sidebar.header("Enter Your Health Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 122, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin Level", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 67.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 0, 120, 30)

# Prediction button
if st.sidebar.button("Predict"):
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes. Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Diabetes. Maintain a healthy lifestyle.")

st.info("ℹ️ This demo version does not include nearby doctor search. Coming soon!")

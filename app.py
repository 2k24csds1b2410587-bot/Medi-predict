import streamlit as st
import joblib
import numpy as np
from openai import OpenAI
import requests

# ================== Load ML Model ==================
model = joblib.load("diabetes_model.pkl")  # make sure file is in health_app/

# ================== Load API Keys ==================
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
LOCATIONIQ_KEY = st.secrets["LOCATIONIQ_KEY"]

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ================== UI ==================
st.set_page_config(page_title="MediPredict", page_icon="🩺", layout="wide")
st.title("🩺 MediPredict - Your AI Health Assistant")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🔮 Diabetes Prediction", "🥗 Weekly Health Plan", "👨‍⚕️ Find Nearby Doctors"])

# ================== 1. Diabetes Prediction ==================
if page == "🔮 Diabetes Prediction":
    st.header("Diabetes Risk Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 25)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes!")
        else:
            st.success("✅ Low Risk of Diabetes!")

# ================== 2. Weekly Health Plan (AI) ==================
elif page == "🥗 Weekly Health Plan":
    st.header("AI-Generated 7-Day Diet & Exercise Plan")

    user_input = st.text_area("Enter your details (age, weight, goals, health conditions, diet preference):")

    if st.button("Generate Plan"):
        if user_input:
            with st.spinner("Generating your weekly plan..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # use "gpt-5" if available
                    messages=[
                        {"role": "system", "content": "You are a professional dietician and fitness coach."},
                        {"role": "user", "content": f"Create a 7-day diet and exercise plan for: {user_input}. Include meals, workouts, and health tips."}
                    ]
                )
                plan = response.choices[0].message.content
            st.subheader("📅 Your Weekly Plan")
            st.write(plan)

# ================== 3. Find Nearby Doctors ==================
elif page == "👨‍⚕️ Find Nearby Doctors":
    st.header("Find Nearby Doctors")

    address = st.text_input("Enter your city or address:")

    if st.button("Search Doctors"):
        if LOCATIONIQ_KEY and address:
            url = f"https://us1.locationiq.com/v1/search.php?key={LOCATIONIQ_KEY}&q={address}&format=json"
            res = requests.get(url).json()

            if isinstance(res, list) and len(res) > 0:
                lat, lon = res[0]["lat"], res[0]["lon"]
                st.write(f"📍 Location: {lat}, {lon}")
                st.info("Nearby doctor search feature can be integrated with Google Places or other API.")
            else:
                st.error("Location not found!")
        else:
            st.error("⚠️ LocationIQ API key missing or invalid.")

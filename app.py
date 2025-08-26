import streamlit as st
import joblib
import numpy as np
import requests
from pathlib import Path
import folium
from streamlit_folium import st_folium

# ====== Load External CSS ======
css_file = Path("style.css")
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("MEDI-PREDICT 🩺")
st.markdown("Enter your health details to predict diabetes risk and chat with your AI health coach.")

# ====== Load ML Model ======
model = joblib.load("diabetes_model.pkl")

# ====== Collapsible User Input Section ======
with st.expander("🩺 Enter Your Health Details"):
    preg = st.slider("Number of pregnancies", 0, 20, 0)
    glucose = st.slider("Glucose level (mg/dL)", 50, 300, 100)
    bp = st.slider("Blood pressure (mmHg)", 40, 200, 70)
    skin = st.slider("Skin thickness (mm)", 0, 100, 20)
    insulin = st.slider("Insulin level (µU/mL)", 0, 900, 80)
    bmi = st.slider("BMI", 10.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
    age = st.slider("Age", 1, 120, 30)

# ====== City Input ======
city = st.text_input("🏙️ Enter your city to find nearby doctors:")

API_KEY = st.secrets.get("LOCATIONIQ_KEY", "")

@st.cache_data(ttl=3600)
def get_nearby_doctors(city):
    try:
        if not API_KEY:
            return [], None, None
        geo_url = f"https://us1.locationiq.com/v1/search.php?key={API_KEY}&q={city}&format=json"
        geo_data = requests.get(geo_url, timeout=10).json()
        if not geo_data:
            return [], None, None
        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])
        places_url = f"https://us1.locationiq.com/v1/nearby.php?key={API_KEY}&lat={lat}&lon={lon}&tag=doctors&radius=3000&format=json"
        places_data = requests.get(places_url, timeout=10).json()
        results = []
        if isinstance(places_data, list):
            for place in places_data[:5]:
                name = place.get("name", "Doctor/Clinic")
                plat = float(place["lat"])
                plon = float(place["lon"])
                results.append((name, plat, plon))
        return results, lat, lon
    except Exception as e:
        return [], None, None

# ====== Risk Check Function ======
def check_risk(glucose, insulin, bp, bmi, age):
    messages = []
    if glucose > 140:
        messages.append("⚠️ High glucose! Reduce sugar intake and consult a doctor.")
    if insulin > 200:
        messages.append("⚠️ High insulin level! Could indicate insulin resistance.")
    if bp > 130:
        messages.append("⚠️ High blood pressure! Monitor your heart health.")
    if bmi > 30:
        messages.append("⚠️ High BMI! Focus on diet and exercise.")
    if age > 50:
        messages.append("⚠️ Age above 50! Regular check-ups recommended.")
    if not messages:
        messages.append("✅ All values are within normal range.")
    return messages

# ====== Rule-Based AI Chatbot ======
def ai_chatbot(user_question, glucose, insulin, bmi, bp, age):
    user_question = user_question.lower()
    response = "🤖 Here's your advice: "

    if "diet" in user_question or "eat" in user_question:
        if glucose > 140:
            response += "Focus on low-sugar foods, whole grains, and vegetables. Avoid refined carbs. "
        elif bmi > 30:
            response += "Eat high-fiber foods and control portion sizes. "
        else:
            response += "Maintain a balanced diet with proteins, carbs, and healthy fats. "
    
    if "exercise" in user_question or "workout" in user_question:
        if bmi > 25 or glucose > 140:
            response += "Exercise at least 30 mins daily: walking, yoga, or cycling. "
        else:
            response += "Keep up your regular physical activity. "
    
    if "blood pressure" in user_question or "bp" in user_question:
        if bp > 130:
            response += "Reduce salt intake, monitor BP regularly, and manage stress. "
        else:
            response += "Your blood pressure is normal, maintain healthy habits. "

    if "insulin" in user_question or "glucose" in user_question:
        if insulin > 200 or glucose > 140:
            response += "Check sugar levels frequently and follow dietary advice. "

    if response.strip() == "🤖 Here's your advice:":
        response += "Could you please specify if you want diet, exercise, or health tips?"

    return response

# ====== Personalized Diet & Lifestyle Plan ======
def get_personalized_plan(glucose, insulin, bp, bmi, age):
    diet_plan = []
    lifestyle = []
    if glucose > 140 or insulin > 200:
        diet_plan.append("🍎 Low-glycemic index foods, whole grains, avoid refined sugar.")
    else:
        diet_plan.append("🥗 Balanced diet with proteins, carbs, and healthy fats.")
    if bmi > 25:
        diet_plan.append("🥦 Include high-fiber vegetables and control portion sizes.")
    if bp > 130:
        diet_plan.append("🥬 Reduce salt intake and avoid processed foods.")
    if bmi > 25 or glucose > 140:
        lifestyle.append("🏃‍♂️ Exercise at least 30 mins daily: walking, cycling, or yoga.")
    if age > 50:
        lifestyle.append("🩺 Schedule regular health check-ups every 6-12 months.")
    if bp > 130:
        lifestyle.append("💤 Ensure 7-8 hours of sleep and manage stress.")
    if not diet_plan:
        diet_plan.append("✅ Your diet is on track!")
    if not lifestyle:
        lifestyle.append("✅ Your lifestyle is healthy!")
    return diet_plan, lifestyle

# ====== Initialize Session State for Chat History ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Prediction & Plan ======
if st.button("Check Risk & Get Personalized Plan"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    st.markdown(f"### Probability of Diabetes: {probability:.2f}")
    st.progress(int(probability*100))
    if prediction == 1:
        st.error("🚨 High Risk of Diabetes!")
    else:
        st.success("😊 Low Risk of Diabetes.")

    # Risk warnings
    feedback = check_risk(glucose, insulin, bp, bmi, age)
    for msg in feedback:
        st.warning(msg)

    # Personalized plan
    diet, life = get_personalized_plan(glucose, insulin, bp, bmi, age)
    st.subheader("🥗 Personalized Diet Recommendations:")
    for item in diet:
        st.markdown(f"- {item}")
    st.subheader("🏋️ Lifestyle Recommendations:")
    for item in life:
        st.markdown(f"- {item}")

# ====== Interactive Multi-Turn Chat ======
st.subheader("💬 Chat with Your AI Health Coach:")
user_question = st.text_input("Ask a question (diet, exercise, health tips):")

if user_question:
    answer = ai_chatbot(user_question, glucose, insulin, bmi, bp, age)
    # Save Q&A in chat history
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Coach", answer))

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.info(f"**{speaker}:** {text}")

# ====== Nearby Doctors ======
if city:
    doctors, lat, lon = get_nearby_doctors(city)
    if doctors:
        st.subheader("👨‍⚕️ Nearby Doctors:")
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], tooltip="Your Location", icon=folium.Icon(color="red")).add_to(m)
        for name, plat, plon in doctors:
            folium.Marker([plat, plon], tooltip=name,
                          popup=f"{name}\nLat: {plat}, Lon: {plon}",
                          icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        st_folium(m, width=700, height=500)
        for name, plat, plon in doctors:
            st.markdown(f"🏥 **{name}**  \n📍 Lat: {plat}, Lon: {plon}")
            st.markdown("---")
    else:
        st.warning("No doctors found or API issue.")

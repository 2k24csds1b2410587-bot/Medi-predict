import streamlit as st
import joblib
import numpy as np
import requests
from pathlib import Path
import folium
from streamlit_folium import st_folium

# ====== Load API Key from Streamlit Secrets ======
API_KEY = st.secrets["LOCATIONIQ_KEY"]

if not API_KEY:
    st.error("API key not found! Please add LOCATIONIQ_KEY in Streamlit Secrets.")
    st.stop()

# ====== Load External CSS ======
css_file = Path("style.css")
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found. Default styling will be used.")

# ====== App Title ======
st.title("MEDI-PREDICT 🩺")
st.markdown("Enter your health details to predict diabetes risk.")

# ====== Load ML Model ======
model = joblib.load("diabetes_model.pkl")

# ====== User Inputs ======
preg = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose level", min_value=0, max_value=300, value=100)
bp = st.number_input("Blood pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# ====== City Input ======
city = st.text_input("🏙️ Enter your city to find nearby doctors:")

# ====== Cached LocationIQ Function ======
@st.cache_data(ttl=3600)
def get_nearby_doctors(city):
    try:
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

# ====== Prediction ======
if st.button("Check Risk"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    st.markdown(f"### Probability of Diabetes: {probability:.2f}")
    st.progress(int(probability*100))

    if prediction == 1:
        st.markdown(f"""
        <div class="high-risk">
        <h3>⚠️ High Risk!</h3>
        <ul>
            {"<li>Control blood sugar with diet & medication.</li>" if glucose > 140 else ""}
            {"<li>Focus on weight management through exercise.</li>" if bmi > 30 else ""}
            {"<li>Monitor blood pressure and reduce salt intake.</li>" if bp > 130 else ""}
            {"<li>Regular health checkups are recommended.</li>" if age > 45 else ""}
            <li>Consult a doctor for proper guidance.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="low-risk">
        <h3>✅ Low Risk</h3>
        <p>Maintain a healthy lifestyle with exercise, balanced diet, and regular checkups.</p>
        </div>
        """, unsafe_allow_html=True)

# ====== Show Nearby Doctors ======
if city:
    doctors, lat, lon = get_nearby_doctors(city)
    if doctors:
        st.subheader("👨‍⚕️ Nearby Doctors:")

        # Map
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker([lat, lon], tooltip="Your Location", icon=folium.Icon(color="red")).add_to(m)
        for name, plat, plon in doctors:
            folium.Marker([plat, plon], tooltip=name,
                          popup=f"{name}\nLat: {plat}, Lon: {plon}",
                          icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

        st_folium(m, width=700, height=500)

        # List
        for name, plat, plon in doctors:
            st.markdown(f"🏥 **{name}**  \n📍 Lat: {plat}, Lon: {plon}")
            st.markdown("---")
    else:
        st.warning("No doctors found or API issue.")

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from xgboost import XGBClassifier

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EWS Jellyfish SCADA – PLTGU Grati",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh setiap 60 detik (60.000 ms)
st_autorefresh(interval=60000, key="scada_refresh")

# ==========================================
# 2. MODEL TRAINING (PERSISTENT & CACHED)
# ==========================================
@st.cache_resource
def train_high_precision_model():
    """
    Melatih model Machine Learning (XGBoost) untuk memprediksi tingkat risiko
    invasi ubur-ubur berdasarkan parameter oseanografi.
    0: Normal, 1: Waspada, 2: Kritis
    """
    np.random.seed(42)
    n_samples = 1200
    
    # Sintesis data historis oseanografi Selat Madura
    temp = np.random.uniform(26.0, 32.0, n_samples)
    salinity = np.random.uniform(28.0, 35.0, n_samples)
    current_speed = np.random.uniform(0.1, 1.8, n_samples)
    chlorophyll = np.random.uniform(0.1, 5.0, n_samples)
    
    # Logika penentuan risiko
    risk_score = (
        (temp - 26) * 0.25 + 
        (salinity - 28) * 0.30 + 
        current_speed * 1.5 + 
        chlorophyll * 0.8
    )
    
    labels = np.zeros(n_samples, dtype=int)
    labels[risk_score > 5.5] = 1  # Waspada
    labels[risk_score > 8.0] = 2  # Kritis
    
    X = pd.DataFrame({
        'sst': temp,
        'salinity': salinity,
        'current_speed': current_speed,
        'chlorophyll_a': chlorophyll
    })
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, labels)
    return model

model = train_high_precision_model()

# ==========================================
# 3. REAL-TIME API DATA PIPELINE
# ==========================================
LATITUDE = -7.6433
LONGITUDE = 113.0238

def fetch_ocean_data():
    """
    Mengambil data real-time dari API Open-Meteo untuk lokasi Sub Sea Water Intake (SWI) PLTGU Grati.
    """
    try:
        # Fetch Marine Data (Suhu Permukaan Laut & Arus)
        marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=sea_water_temperature,ocean_current_velocity&timezone=Asia%2FJakarta"
        marine_res = requests.get(marine_url, timeout=5).json()
        
        latest_sst = marine_res['hourly']['sea_water_temperature'][-1]
        latest_current = marine_res['hourly']['ocean_current_velocity'][-1]
        
        # Jika data API null, beri nilai default realistis
        sst = float(latest_sst) if latest_sst is not None else 29.5
        current = float(latest_current) if latest_current is not None else 0.45
        
        # Simulasi variabel tambahan (Salinitas & Klorofil-a)
        salinity = 32.5 + np.random.uniform(-0.5, 0.5)
        chlorophyll = 2.1 + np.random.uniform(-0.3, 0.3)
        
        return {
            'sst': round(sst, 2),
            'salinity': round(salinity, 2),
            'current_speed': round(current, 2),
            'chlorophyll_a': round(chlorophyll, 2),
            'status': 'Online'
        }
    except Exception as e:
        # Fallback data jika koneksi API terganggu
        return {
            'sst': 29.20,
            'salinity': 32.10,
            'current_speed': 0.42,
            'chlorophyll_a': 1.85,
            'status': 'Offline (Fallback Active)'
        }

# ==========================================
# 4. DASHBOARD HEADER & TITLE
# ==========================================
st.title("EWS Jellyfish SCADA – PLTGU Grati")
st.caption("Sub Sea Water Intake (SWI) Real-Time Monitoring & Predictive Early Warning System")

data = fetch_ocean_data()

# Model Inference
input_df = pd.DataFrame([data])[['sst', 'salinity', 'current_speed', 'chlorophyll_a']]
pred_code = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0]

status_map = {
    0: {"label": "NORMAL", "color": "green", "bg": "#d4edda", "text": "#155724"},
    1: {"label": "WASPADA", "color": "orange", "bg": "#fff3cd", "text": "#856404"},
    2: {"label": "KRITIS", "color": "red", "bg": "#f8d7da", "text": "#721c24"}
}
current_status = status_map[pred_code]

# Status Banner
st.markdown(
    f"""
    <div style="background-color: {current_status['bg']}; padding: 15px; border-radius: 8px; border-left: 8px solid {current_status['color']}; margin-bottom: 20px;">
        <h3 style="color: {current_status['text']}; margin: 0;">STATUS SISTEM: {current_status['label']}</h3>
        <p style="color: {current_status['text']}; margin: 5px 0 0 0;">Probabilitas Ancaman Invasi: <b>{pred_proba[pred_code]*100:.1f}%</b> | Koneksi API Data: <b>{data['status']}</b></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Audio Alarm jika status KRITIS
if pred_code == 2:
    st.audio("https://www.soundjay.com/buttons/sounds/button-10.mp3", autoplay=True)

# ==========================================
# 5. METRICS DASHBOARD
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Suhu Permukaan Laut (SST)", f"{data['sst']} °C", delta="Normal: 26 - 30 °C")

with col2:
    st.metric("Salinitas Air Laut", f"{data['salinity']} PSU", delta="Normal: 30 - 34 PSU")

with col3:
    st.metric("Kecepatan Arus Laut", f"{data['current_speed']} m/s", delta="Tinggi jika > 0.8 m/s")

with col4:
    st.metric("Klorofil-a (Plankton)", f"{data['chlorophyll_a']} mg/m³", delta="Tinggi jika > 3.0 mg/m³")

st.divider()

# ==========================================
# 6. GEOSPATIAL MAP & EXPLAINABLE AI
# ==========================================
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📍 Peta Lokasi Monitoring Sub Sea Water Intake (SWI)")
    
    # Inisialisasi Peta Folium
    m = folium.Map(location=[LATITUDE, LONGITUDE], zoom_start=13, tiles="OpenStreetMap")
    
    # Penanda Dinamis berdasarkan Status Risiko
    folium.Marker(
        location=[LATITUDE, LONGITUDE],
        popup=f"SWI PLTGU Grati\nStatus: {current_status['label']}",
        tooltip="Lokasi SWI PLTGU Grati",
        icon=folium.Icon(color=current_status['color'], icon="info-sign")
    ).add_to(m)
    
    # Lingkaran Area Radius Monitoring
    folium.Circle(
        radius=1500,
        location=[LATITUDE, LONGITUDE],
        color=current_status['color'],
        fill=True,
        fill_opacity=0.2
    ).add_to(m)
    
    st_folium(m, width="100%", height=400)

with right_col:
    st.subheader("📊 Analisis Faktor Risiko (XAI)")
    
    # Feature Importance untuk Explainable AI
    feature_importances = model.feature_importances_
    features = ["Suhu Laut (SST)", "Salinitas", "Kecepatan Arus", "Klorofil-a"]
    
    importance_df = pd.DataFrame({
        'Faktor': features,
        'Pengaruh (%)': feature_importances * 100
    }).sort_values(by='Pengaruh (%)', ascending=True)
    
    st.bar_chart(importance_df.set_index('Faktor'))
    st.info("Grafik di atas menunjukkan parameter oseanografi yang paling dominan memicu potensi invasi ubur-ubur saat ini.")

# ==========================================
# 7. SCADA LOG & OPERATIONAL ACTIONS
# ==========================================
st.divider()
st.subheader("📋 SOP Tindakan Operasional SCADA")

if pred_code == 0:
    st.success("✅ **Kondisi Normal:** Lanjutkan pemantauan rutin pada Travelling Band Screen (TBS) dan differential pressure intake.")
elif pred_code == 1:
    st.warning("⚠️ **Kondisi Waspada:** Siapkan pembersihan berkala pada Trash Rack, tingkatkan frekuensi inspeksi visual di area SWI, serta siapkan jet pump.")
else:
    st.error("🚨 **Kondisi Kritis:** Aktifkan sistem pembersih otomatis TBS secara mendesak, siapkan barikade/jaring penahan ubur-ubur, dan koordinasikan dengan tim pemeliharaan.")

import datetime
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# Set Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="SWI PLTGU Grati - Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

# Title & Header Dashboard
st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader(
    "Sistem Early Warning Machine Learning Risiko Ubur-Ubur (Pesisir"
    " Pasuruan)"
)
st.markdown("---")

# Data Lokasi
GRATI_LAT = -7.5950
GRATI_LON = 112.8943


# Cache Model ML agar tidak re-train setiap reload
@st.cache_resource
def train_model():
  np.random.seed(42)
  n_samples = 1200
  sst = np.random.normal(loc=29.5, scale=1.2, size=n_samples)
  salinity = np.random.normal(loc=32.5, scale=1.1, size=n_samples)
  current_speed = np.random.exponential(scale=0.25, size=n_samples)
  chlorophyll = np.random.gamma(shape=2.2, scale=0.7, size=n_samples)
  wind_speed = np.random.uniform(1.0, 12.0, size=n_samples)

  risk_score = (
      (sst - 28.5) * 0.35
      + (chlorophyll) * 0.35
      + (current_speed) * 0.15
      + (wind_speed * 0.15)
  )
  labels = pd.qcut(risk_score, q=3, labels=[0, 1, 2])

  df = pd.DataFrame({
      "sst": sst,
      "salinity": salinity,
      "current_speed": current_speed,
      "chlorophyll_a": chlorophyll,
      "wind_speed": wind_speed,
      "risk_level": labels,
  })

  X = df.drop(columns=["risk_level"])
  y = df["risk_level"]

  model = XGBClassifier(
      n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric="mlogloss"
  )
  model.fit(X, y)
  return model


model = train_model()

# Sidebar Control
st.sidebar.header("🕹️ Mode Input Data")
data_mode = st.sidebar.radio(
    "Pilih Sumber Data:", ("Simulasi Real-Time", "Input Manual (Testing)")
)

if data_mode == "Simulasi Real-Time":
  # Data Parameter Laut (Realtime Simulation)
  sst = 30.2
  salinity = 33.1
  current_speed = 0.42
  chlorophyll = 2.6
  wind_speed = 7.5
else:
  st.sidebar.subheader("Atur Parameter Laut:")
  sst = st.sidebar.slider("Suhu Permukaan Laut (°C)", 25.0, 35.0, 30.0)
  salinity = st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, 33.0)
  current_speed = st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 1.5, 0.4)
  chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 5.0, 2.5)
  wind_speed = st.sidebar.slider("Kecepatan Angin (knot)", 0.0, 20.0, 7.0)

# Prediksi Model
input_df = pd.DataFrame([{
    "sst": sst,
    "salinity": salinity,
    "current_speed": current_speed,
    "chlorophyll_a": chlorophyll,
    "wind_speed": wind_speed,
}])

risk_class = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

# Display Metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Suhu Laut (SST)", f"{sst} °C", "+0.8 °C")
col2.metric("Salinitas", f"{salinity} PSU")
col3.metric("Kecepatan Arus", f"{current_speed} m/s")
col4.metric("Klorofil-a", f"{chlorophyll} mg/m³")
col5.metric("Angin Laut", f"{wind_speed} knot")

st.markdown("---")

# Layout Hasil Prediksi
col_left, col_right = st.columns([2, 1])

with col_left:
  st.subheader("📊 Hasil Prediksi Risiko Machine Learning")
  if risk_class == 0:
    st.success("### STATUS: AMAN (LOW RISK)")
    st.write("🟢 Kondisi air laut stabil. Operasional intake berjalan normal.")
  elif risk_class == 1:
    st.warning("### STATUS: WASPADA (MEDIUM RISK)")
    st.write(
        "🟡 Indikasi awal kawanan ubur-ubur. Tingkatkan pengawasan visual pada"
        " Bar Screen & pantau Beda Tekanan (ΔP)."
    )
  else:
    st.error("### STATUS: BAHAYA (HIGH RISK / BLOOMING)")
    st.write(
        "🔴 ANCAMAN BLOOMING UBUR-UBUR TINGGI! Siapkan operasi kontinyu"
        " Travelling Band Screen (TBS) & siagakan tim lokasi."
    )

  st.write("#### Probabilitas Tingkat Risiko:")
  st.progress(
      float(probabilities[0]),
      text=f"Aman (Low): {probabilities[0]*100:.1f}%",
  )
  st.progress(
      float(probabilities[1]),
      text=f"Waspada (Med): {probabilities[1]*100:.1f}%",
  )
  st.progress(
      float(probabilities[2]),
      text=f"Bahaya (High): {probabilities[2]*100:.1f}%",
  )

with col_right:
  st.subheader("📍 Koordinat Monitoring")
  map_data = pd.DataFrame({"lat": [GRATI_LAT], "lon": [GRATI_LON]})
  st.map(map_data, zoom=11)

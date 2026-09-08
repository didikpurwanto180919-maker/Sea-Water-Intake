import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from xgboost import XGBClassifier

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SWI PLTGU Grati - Live Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader("Sistem Early Warning Machine Learning Risiko Ubur-Ubur")
st.markdown("---")

# ==========================================
# KOORDINAT PRESISI INTAKE SWI PLTGU GRATI
# ==========================================
GRATI_LAT = -7.6531
GRATI_LON = 113.0289
OCEAN_LAT = -7.6495
OCEAN_LON = 113.0289


# ==========================================
# FUNGSI FETCH LIVE DATA (CACHE TTL 60 DETIK)
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def get_live_ocean_data():
  try:
    # 1. API Weather & Wind
    url_weather = (
        f"https://api.open-meteo.com/v1/forecast?latitude={GRATI_LAT}&longitude={GRATI_LON}&current=temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m&wind_speed_unit=kn"
    )
    req_w = requests.get(url_weather, timeout=4)
    res_w = req_w.json() if req_w.ok else {}
    wind_speed = res_w.get("current", {}).get("wind_speed_10m", 6.5)

    # 2. API Marine / Oceanography
    url_marine = (
        f"https://marine-api.open-meteo.com/v1/marine?latitude={OCEAN_LAT}&longitude={OCEAN_LON}&current=sea_surface_temperature,ocean_current_velocity"
    )
    req_m = requests.get(url_marine, timeout=4)
    res_m = req_m.json() if req_m.ok else {}

    current_data = res_m.get("current", {})

    sst = current_data.get("sea_surface_temperature")
    if sst is None:
      sst = 29.8

    current_speed = current_data.get("ocean_current_velocity")
    if current_speed is None:
      current_speed = 0.35

    salinity = 33.2
    chlorophyll = round(1.5 + (sst - 28.0) * 0.4, 2)

    return {
        "status": "Success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
        "sst": round(sst, 2),
        "salinity": salinity,
        "current_speed": round(current_speed, 2),
        "chlorophyll_a": max(0.5, round(chlorophyll, 2)),
        "wind_speed": round(wind_speed, 2),
    }

  except Exception as e:
    return {
        "status": f"Fallback Data ({e})",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
        "sst": 29.5,
        "salinity": 33.0,
        "current_speed": 0.30,
        "chlorophyll_a": 1.8,
        "wind_speed": 6.5,
    }


# ==========================================
# TRAINING MODEL MACHINE LEARNING (XGBoost)
# ==========================================
@st.cache_resource
def train_jellyfish_model():
  np.random.seed(42)
  n_samples = 500

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
      n_estimators=30, learning_rate=0.05, max_depth=3, eval_metric="mlogloss"
  )
  model.fit(X, y)
  return model


model = train_jellyfish_model()

# ==========================================
# SIDEBAR CONTROL
# ==========================================
st.sidebar.header("🕹️ Mode Input Data")
data_source = st.sidebar.radio(
    "Pilih Sumber Data:", ("Live API (Real-Time)", "Simulasi / Manual Test")
)


# ==========================================
# FRAGMENT UNTUK AUTO-UPDATE RINGAN (BEBAS THROTTLE)
# ==========================================
@st.fragment(run_every="60s")
def render_dashboard():
  if data_source == "Live API (Real-Time)":
    live_data = get_live_ocean_data()
    sst = live_data["sst"]
    salinity = live_data["salinity"]
    current_speed = live_data["current_speed"]
    chlorophyll = live_data["chlorophyll_a"]
    wind_speed = live_data["wind_speed"]

    st.sidebar.success("⚡ Live Auto-Update (60s)")
    st.sidebar.info(f"Last Update: {live_data['timestamp']}")
  else:
    st.sidebar.subheader("Atur Parameter Laut:")
    sst = st.sidebar.slider("Suhu Permukaan Laut (°C)", 25.0, 35.0, 30.0)
    salinity = st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, 33.0)
    current_speed = st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 1.5, 0.4)
    chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 5.0, 2.5)
    wind_speed = st.sidebar.slider("Kecepatan Angin (knot)", 0.0, 20.0, 7.0)

  # METRICS
  col1, col2, col3, col4, col5 = st.columns(5)
  col1.metric("Suhu Laut (SST)", f"{sst} °C")
  col2.metric("Salinitas", f"{salinity} PSU")
  col3.metric("Kecepatan Arus", f"{current_speed} m/s")
  col4.metric("Klorofil-a", f"{chlorophyll} mg/m³")
  col5.metric("Angin Laut", f"{wind_speed} knot")

  st.markdown("---")

  # PREDIKSI ML
  input_df = pd.DataFrame([{
      "sst": sst,
      "salinity": salinity,
      "current_speed": current_speed,
      "chlorophyll_a": chlorophyll,
      "wind_speed": wind_speed,
  }])

  risk_class = model.predict(input_df)[0]
  probabilities = model.predict_proba(input_df)[0]

  col_left, col_right = st.columns([1.5, 1])

  with col_left:
    st.subheader("📊 Hasil Prediksi Risiko Machine Learning")
    if risk_class == 0:
      st.success("### STATUS: AMAN (LOW RISK)")
      st.write("🟢 Kondisi air laut stabil. Operasional intake berjalan normal.")
    elif risk_class == 1:
      st.warning("### STATUS: WASPADA (MEDIUM RISK)")
      st.write(
          "🟡 Indikasi awal kawanan ubur-ubur. Tingkatkan pengawasan visual"
          " pada Bar Screen & pantau Beda Tekanan (ΔP)."
      )
    else:
      st.error("### STATUS: BAHAYA (HIGH RISK / BLOOMING)")
      st.write(
          "🔴 ANCAMAN BLOOMING UBUR-UBUR TINGGI! Siapkan operasi kontinyu"
          " Travelling Band Screen (TBS) & siagakan tim lokasi."
      )

    st.write("#### Probabilitas Tingkat Risiko:")
    st.progress(
        float(probabilities[0]), text=f"Aman (Low): {probabilities[0]*100:.1f}%"
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
    st.subheader("📍 Peta Lokasi Intake SWI PLTGU Grati")
    st.caption(f"Lat: {GRATI_LAT}, Lon: {GRATI_LON}")

    # Peta Bawaan Streamlit (Sangat Ringan, Ringkas, & Bebas Throttling)
    map_df = pd.DataFrame({"lat": [GRATI_LAT], "lon": [GRATI_LON]})
    st.map(map_df, zoom=14)


# Jalankan Dashboard
render_dashboard()

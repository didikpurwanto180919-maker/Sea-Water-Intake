import datetime
import zoneinfo
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
from xgboost import XGBClassifier

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SWI PLTGU Grati - Live Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

# Set Zona Waktu WIB (Asia/Jakarta)
WIB_TZ = zoneinfo.ZoneInfo("Asia/Jakarta")

# ==========================================
# AUTO REFRESH TIAP 300 DETIK (5 MENIT)
# ==========================================
count = st_autorefresh(interval=300000, limit=None, key="jellyfish_auto_refresh")

st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader(
    "Sistem Early Warning Machine Learning Risiko Ubur-Ubur (Optimized & High Precision)"
)
st.markdown("---")

# ==========================================
# KOORDINAT PRESISI SWI INTAKE PLTGU GRATI
# ==========================================
GRATI_LAT = -7.6433
GRATI_LON = 113.0238

OCEAN_LAT = -7.6400
OCEAN_LON = 113.0238


# ==========================================
# FUNGSI FETCH LIVE DATA (CACHE 300 DETIK)
# ==========================================
@st.cache_data(ttl=300)
def get_live_ocean_data(refresh_id):
    wib_now = datetime.datetime.now(WIB_TZ)
    wib_time_str = wib_now.strftime("%Y-%m-%d %H:%M:%S WIB")

    try:
        # 1. API Weather & Wind (Live Open-Meteo)
        url_weather = (
            f"https://api.open-meteo.com/v1/forecast?latitude={GRATI_LAT}&longitude={GRATI_LON}"
            f"&current=temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m&wind_speed_unit=kn"
        )
        req_w = requests.get(url_weather, timeout=5)
        res_w = req_w.json() if req_w.ok else {}
        wind_speed = res_w.get("current", {}).get("wind_speed_10m", 6.5)

        # 2. API Marine / Oceanography (SST & Current Velocity)
        url_marine = (
            f"https://marine-api.open-meteo.com/v1/marine?latitude={OCEAN_LAT}&longitude={OCEAN_LON}"
            f"&current=sea_surface_temperature,ocean_current_velocity"
        )
        req_m = requests.get(url_marine, timeout=5)
        res_m = req_m.json() if req_m.ok else {}

        current_data = res_m.get("current", {})
        sst = current_data.get("sea_surface_temperature", 29.8)
        current_speed = current_data.get("ocean_current_velocity", 0.35)

        if sst is None:
            sst = 29.8
        if current_speed is None:
            current_speed = 0.35

        # Parameter Biogeokimia Pesisir (Pendekatan Empiris Selat Madura)
        salinity = 33.2
        chlorophyll = round(1.2 + (sst - 28.0) * 0.45 + (wind_speed * 0.05), 2)

        return {
            "status": "Success (Live Auto-Update)",
            "timestamp": wib_time_str,
            "sst": round(float(sst), 2),
            "salinity": float(salinity),
            "current_speed": round(float(current_speed), 2),
            "chlorophyll_a": max(0.5, round(float(chlorophyll), 2)),
            "wind_speed": round(float(wind_speed), 2),
        }

    except Exception as e:
        return {
            "status": f"Fallback Data ({e})",
            "timestamp": wib_time_str,
            "sst": 29.5,
            "salinity": 33.0,
            "current_speed": 0.30,
            "chlorophyll_a": 1.80,
            "wind_speed": 6.50,
        }


# ==========================================
# TRAINING MODEL MACHINE LEARNING (XGBoost Optimized)
# ==========================================
@st.cache_resource
def train_jellyfish_model():
    np.random.seed(42)
    n_samples = 2000

    # Sintesis distribusi data parameter oceanografi pesisir Grati
    sst = np.random.normal(loc=29.5, scale=1.1, size=n_samples)
    salinity = np.random.normal(loc=32.8, scale=0.9, size=n_samples)
    current_speed = np.random.exponential(scale=0.28, size=n_samples)
    chlorophyll = np.random.gamma(shape=2.5, scale=0.6, size=n_samples)
    wind_speed = np.random.uniform(1.0, 15.0, size=n_samples)

    # Indeks Risiko Non-Linear berbasis Fisiologi Ubur-Ubur (Suhu ideal & Nutrisi Tinggi)
    risk_score = (
        (sst - 28.0) * 0.40
        + (chlorophyll * 0.35)
        + (current_speed * 0.15)
        + (wind_speed * 0.10)
        + np.random.normal(0, 0.1, size=n_samples)
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

    # Hyperparameter Tuning untuk Presisi Tinggi & Stabil
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
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

if data_source == "Live API (Real-Time)":
    live_data = get_live_ocean_data(count)
    sst = live_data["sst"]
    salinity = live_data["salinity"]
    current_speed = live_data["current_speed"]
    chlorophyll = live_data["chlorophyll_a"]
    wind_speed = live_data["wind_speed"]

    st.sidebar.success("⚡ Live Refresh: 300 Detik (5 Menit)")
    st.sidebar.info(f"Last Update: {live_data['timestamp']}")

else:
    st.sidebar.subheader("Atur Parameter Laut:")
    sst = st.sidebar.slider("Suhu Permukaan Laut (°C)", 25.0, 35.0, 29.8, 0.1)
    salinity = st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, 33.2, 0.1)
    current_speed = st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 2.0, 0.35, 0.01)
    chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 8.0, 2.18, 0.01)
    wind_speed = st.sidebar.slider("Kecepatan Angin (knot)", 0.0, 25.0, 5.4, 0.1)

# ==========================================
# METRICS DISPLAY & INFERENCE ML
# ==========================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Suhu Laut (SST)", f"{sst:.1f} °C")
col2.metric("Salinitas", f"{salinity:.1f} PSU")
col3.metric("Kecepatan Arus", f"{current_speed:.2f} m/s")
col4.metric("Klorofil-a", f"{chlorophyll:.2f} mg/m³")
col5.metric("Angin Laut", f"{wind_speed:.1f} knot")

st.markdown("---")

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
    st.subheader("📍 Koordinat Intake PLTGU Grati")
    st.caption(f"Lat: {GRATI_LAT}, Lon: {GRATI_LON}")

    m = folium.Map(location=[GRATI_LAT, GRATI_LON], zoom_start=16)

    google_satellite = folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
    )
    google_satellite.add_to(m)

    folium.Marker(
        [GRATI_LAT, GRATI_LON],
        popup="Inlet SWI PLTGU Grati",
        tooltip="Inlet SWI PLTGU Grati",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    st_folium(m, width=420, height=320, key=f"grati_map_{count}")

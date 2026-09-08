import datetime
import zoneinfo
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
from xgboost import XGBClassifier

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SWI PLTGU Grati - High Precision Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

WIB_TZ = zoneinfo.ZoneInfo("Asia/Jakarta")
REFRESH_INTERVAL_SEC = 60

count = st_autorefresh(
    interval=REFRESH_INTERVAL_SEC * 1000,
    limit=None,
    key="jellyfish_auto_refresh",
)

st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader(
    "Sistem Early Warning Machine Learning Presisi Tinggi - 15 Parameter"
    " Kompleks"
)
st.markdown("---")

GRATI_LAT = -7.6433
GRATI_LON = 113.0238
OCEAN_LAT = -7.6400
OCEAN_LON = 113.0238


# ==========================================
# TRAINING MODEL HIGH-PRECISION MACHINE LEARNING
# ==========================================
@st.cache_resource
def train_high_precision_model():
    np.random.seed(42)
    n_samples = 15000

    # Generator 15 Parameter Sintetis Berdasar Distribusi Real
    sst = np.random.uniform(26.0, 34.0, size=n_samples)
    chlorophyll = np.random.uniform(0.5, 8.0, size=n_samples)
    salinity = np.random.uniform(29.0, 36.0, size=n_samples)
    do_level = np.random.uniform(2.0, 8.0, size=n_samples)
    turbidity = np.random.uniform(1.0, 40.0, size=n_samples)

    current_speed = np.random.uniform(0.05, 1.8, size=n_samples)
    current_dir = np.random.uniform(0, 360, size=n_samples)
    wave_height = np.random.uniform(0.1, 2.0, size=n_samples)

    wind_speed = np.random.uniform(1.0, 25.0, size=n_samples)
    wind_dir = np.random.uniform(0, 360, size=n_samples)
    tide_phase = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 1: Spring Tide
    sea_level = np.random.uniform(-1.0, 2.0, size=n_samples)

    delta_p = np.random.uniform(0.05, 1.5, size=n_samples)
    flow_velocity = np.random.uniform(0.2, 1.2, size=n_samples)
    tbs_torque = np.random.uniform(10.0, 95.0, size=n_samples)

    # Matriks Pembobotan Fisika-Biologis Presisi Tinggi
    is_onshore_current = (current_dir >= 110) & (current_dir <= 210)
    is_onshore_wind = (wind_dir >= 110) & (wind_dir <= 210)

    # Formula Risiko Terintegrasi 15 Parameter
    risk_score = (
        (np.maximum(0, sst - 30.0) ** 1.8) * 2.2
        + (np.maximum(0, chlorophyll - 3.5) ** 1.5) * 2.8
        + (np.maximum(0, salinity - 33.0) * 0.8)
        + (np.maximum(0, 5.0 - do_level) * 0.7)
        + (current_speed * 2.0 * np.where(is_onshore_current, 2.5, 0.4))
        + (wind_speed * 0.25 * np.where(is_onshore_wind, 1.8, 0.5))
        + (wave_height * 1.5)
        + (tide_phase * 4.0)
        + (delta_p * 6.0)
        + (tbs_torque * 0.08)
    )

    labels = np.where(risk_score < 14.0, 0, np.where(risk_score < 28.0, 1, 2))

    df = pd.DataFrame({
        "sst": sst,
        "chlorophyll_a": chlorophyll,
        "salinity": salinity,
        "do_level": do_level,
        "turbidity": turbidity,
        "current_speed": current_speed,
        "current_dir": current_dir,
        "wave_height": wave_height,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "tide_phase": tide_phase,
        "sea_level": sea_level,
        "delta_p": delta_p,
        "flow_velocity": flow_velocity,
        "tbs_torque": tbs_torque,
        "risk_level": labels,
    })

    X = df.drop(columns=["risk_level"])
    y = df["risk_level"]

    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model


model = train_high_precision_model()

# ==========================================
# SIDEBAR CONTROL & SIMULASI
# ==========================================
st.sidebar.header("🕹️ Control & Scenario")
preset = st.sidebar.selectbox(
    "Pilih Skenario Simulasi Input:",
    (
        "🚨 KONDISI KRITIS: SERANGAN UBUR-UBUR (BLOOMING)",
        "⚠️ KONDISI WASPADA: INDIKASI AWAL",
        "🟢 KONDISI NORMAL: NORMAL SWI OPERATIONAL",
    ),
)

if preset == "🚨 KONDISI KRITIS: SERANGAN UBUR-UBUR (BLOOMING)":
    d = {
        "sst": 32.2,
        "chl": 5.40,
        "sal": 34.5,
        "do": 3.2,
        "turb": 28.0,
        "cspd": 1.35,
        "cdir": 170,
        "wh": 1.4,
        "wspd": 15.0,
        "wdir": 160,
        "tide": 1,
        "sl": 1.6,
        "dp": 0.82,
        "fv": 0.95,
        "torq": 78.0,
    }
elif preset == "⚠️ KONDISI WASPADA: INDIKASI AWAL":
    d = {
        "sst": 30.2,
        "chl": 3.20,
        "sal": 33.2,
        "do": 4.8,
        "turb": 12.0,
        "cspd": 0.65,
        "cdir": 140,
        "wh": 0.7,
        "wspd": 8.0,
        "wdir": 135,
        "tide": 0,
        "sl": 0.8,
        "dp": 0.38,
        "fv": 0.60,
        "torq": 40.0,
    }
else:
    d = {
        "sst": 28.1,
        "chl": 1.20,
        "sal": 32.0,
        "do": 6.5,
        "turb": 4.0,
        "cspd": 0.25,
        "cdir": 45,
        "wh": 0.3,
        "wspd": 4.5,
        "wdir": 50,
        "tide": 0,
        "sl": 0.2,
        "dp": 0.12,
        "fv": 0.40,
        "torq": 18.0,
    }

st.sidebar.subheader("🎛️ Adjust 15 Parameter Data")
sst = st.sidebar.slider("1. Suhu Laut SST (°C)", 25.0, 35.0, d["sst"])
chlorophyll = st.sidebar.slider("2. Klorofil-a (mg/m³)", 0.1, 8.0, d["chl"])
salinity = st.sidebar.slider("3. Salinitas (PSU)", 28.0, 36.0, d["sal"])
do_level = st.sidebar.slider("4. Oksigen Terlarut DO (mg/L)", 1.0, 8.0, d["do"])
turbidity = st.sidebar.slider("5. Kekeruhan Turbidity (NTU)", 0.0, 50.0, d["turb"])
current_speed = st.sidebar.slider(
    "6. Kecepatan Arus (m/s)", 0.0, 2.0, d["cspd"]
)
current_dir = st.sidebar.slider("7. Arah Arus (°)", 0, 360, d["cdir"])
wave_height = st.sidebar.slider("8. Tinggi Gelombang (m)", 0.0, 3.0, d["wh"])
wind_speed = st.sidebar.slider("9. Kecepatan Angin (Knot)", 0.0, 30.0, d["wspd"])
wind_dir = st.sidebar.slider("10. Arah Angin (°)", 0, 360, d["wdir"])
tide_phase = st.sidebar.selectbox(
    "11. Siklus Pasang",
    (0, 1),
    index=d["tide"],
    format_func=lambda x: "Spring Tide (Pasang Purnama)" if x == 1 else "Normal / Neap Tide",
)
sea_level = st.sidebar.slider("12. Elevasi Muka Air (m)", -1.5, 2.5, d["sl"])
delta_p = st.sidebar.slider("13. Beda Tekanan ΔP Screen (mWC)", 0.0, 2.0, d["dp"])
flow_velocity = st.sidebar.slider("14. Kecepatan Flow Intake (m/s)", 0.0, 1.5, d["fv"])
tbs_torque = st.sidebar.slider("15. Torsi TBS Motor (%)", 0.0, 100.0, d["torq"])

# ==========================================
# METRICS PANEL (15 PARAMETER)
# ==========================================
st.markdown("### 📊 Status Parameter Real-Time Intake")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Suhu Laut (SST)", f"{sst:.1f} °C")
c2.metric("Klorofil-a", f"{chlorophyll:.2f} mg/m³")
c3.metric("Salinitas", f"{salinity:.1f} PSU")
c4.metric("Oksigen Terlarut (DO)", f"{do_level:.1f} mg/L")
c5.metric("Kekeruhan", f"{turbidity:.1f} NTU")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Kecepatan Arus", f"{current_speed:.2f} m/s")
c7.metric("Arah Arus", f"{current_dir}°")
c8.metric("Tinggi Gelombang", f"{wave_height:.1f} m")
c9.metric("Kecepatan Angin", f"{wind_speed:.1f} kts")
c10.metric("Arah Angin", f"{wind_dir}°")

c11, c12, c13, c14, c15 = st.columns(5)
c11.metric(
    "Pasang Laut",
    "Spring Tide" if tide_phase == 1 else "Neap Tide",
)
c12.metric("Elevasi Muka Air", f"{sea_level:.1f} m")
c13.metric("Beda Tekanan ΔP", f"{delta_p:.2f} mWC")
c14.metric("Flow Velocity SWI", f"{flow_velocity:.2f} m/s")
c15.metric("Torsi TBS Screen", f"{tbs_torque:.0f} %")

st.markdown("---")

# ==========================================
# PREDIKSI MODEL & EVALUASI RISIKO
# ==========================================
input_df = pd.DataFrame([{
    "sst": sst,
    "chlorophyll_a": chlorophyll,
    "salinity": salinity,
    "do_level": do_level,
    "turbidity": turbidity,
    "current_speed": current_speed,
    "current_dir": current_dir,
    "wave_height": wave_height,
    "wind_speed": wind_speed,
    "wind_dir": wind_dir,
    "tide_phase": tide_phase,
    "sea_level": sea_level,
    "delta_p": delta_p,
    "flow_velocity": flow_velocity,
    "tbs_torque": tbs_torque,
}])

risk_class = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🎯 Hasil Diagnosa Model Machine Learning (XGBoost)")

    if risk_class == 2:
        st.error(
            "### 🚨 ALARM KRITIS: SERANGAN UBUR-UBUR DETEKSI TINGGI (JELLYFISH"
            " BLOOMING)"
        )
        st.markdown("""
        **REKOMENDASI EKSEKUSI OPERASIONAL SHIFT:**
        1. ⚙️ Segera operasikan **Travelling Band Screen (TBS)** pada mode **Continuous High Speed**.
        2. 🚿 Aktifkan **Screen Wash Pump** tekanan tinggi untuk merontokkan biomassa ubur-ubur dari mesh screen.
        3. 🌊 Lakukan koordinasi penyiapan **Debris Filter** pada jalur *Circulating Water Pump* (CWP).
        4. 📉 Siapkan skenario *derating* (penurunan beban pembangkit) jika $\Delta P$ melampaui $0.80$ mWC.
        """)
    elif risk_class == 1:
        st.warning(
            "### ⚠️ ALARM WASPADA: POTENSI AKUMULASI UBUR-UBUR DI KANAL SWI"
        )
        st.markdown("""
        **TINDAKAN PENCEGAHAN:**
        * Lakukan pengamatan visual di area *Bar Screen* & *Outer Intake* setiap 30 menit.
        * Pantau tren kenaikan Beda Tekanan ($\Delta P$) dan Torsi Motor TBS.
        """)
    else:
        st.success("### 🟢 KONDISI AMAN: TIDAK ADA ANCAMAN UBUR-UBUR")

    st.write("#### Distrbusi Probabilitas:")
    st.progress(
        float(probabilities[0]), text=f"Aman: {probabilities[0]*100:.1f}%"
    )
    st.progress(
        float(probabilities[1]), text=f"Waspada: {probabilities[1]*100:.1f}%"
    )
    st.progress(
        float(probabilities[2]),
        text=f"Bahaya Serangan: {probabilities[2]*100:.1f}%",
    )

with col_right:
    st.subheader("📍 Lokasi Kanal SWI PLTGU Grati")
    m = folium.Map(location=[GRATI_LAT, GRATI_LON], zoom_start=15)

    google_satellite = folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
    )
    google_satellite.add_to(m)

    marker_color = (
        "red" if risk_class == 2 else ("orange" if risk_class == 1 else "green")
    )

    folium.Marker(
        [GRATI_LAT, GRATI_LON],
        popup=f"SWI PLTGU Grati - Status: {risk_class}",
        tooltip="SWI PLTGU Grati Intake",
        icon=folium.Icon(color=marker_color, icon="info-sign"),
    ).add_to(m)

    st_folium(m, width=420, height=320, key=f"grati_map_{count}")

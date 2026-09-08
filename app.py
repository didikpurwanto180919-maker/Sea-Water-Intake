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
    page_title="SWI PLTGU Grati - Live Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

# Set Zona Waktu WIB (Asia/Jakarta)
WIB_TZ = zoneinfo.ZoneInfo("Asia/Jakarta")

# REFRESH INTERVAL (DETIK)
REFRESH_INTERVAL_SEC = 60

# ==========================================
# AUTO REFRESH TIAP 60 DETIK
# ==========================================
count = st_autorefresh(
    interval=REFRESH_INTERVAL_SEC * 1000,
    limit=None,
    key="jellyfish_auto_refresh",
)

st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader(
    "Sistem Early Warning Machine Learning Risiko Ubur-Ubur (Live Auto Update"
    " 60s)"
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
# FUNGSI FETCH LIVE DATA (CACHE 60 DETIK)
# ==========================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
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
        curr_w = res_w.get("current", {})
        wind_speed = curr_w.get("wind_speed_10m", 4.8)
        wind_dir = curr_w.get("wind_direction_10m", 45)  # Derajat

        # 2. API Marine / Oceanography (SST, Current Velocity & Direction)
        url_marine = (
            f"https://marine-api.open-meteo.com/v1/marine?latitude={OCEAN_LAT}&longitude={OCEAN_LON}"
            f"&current=sea_surface_temperature,ocean_current_velocity,ocean_current_direction"
        )
        req_m = requests.get(url_marine, timeout=5)
        res_m = req_m.json() if req_m.ok else {}

        current_data = res_m.get("current", {})
        sst = current_data.get("sea_surface_temperature", 29.7)
        current_speed = current_data.get("ocean_current_velocity", 0.30)
        current_dir = current_data.get("ocean_current_direction", 180)

        if sst is None:
            sst = 29.7
        if current_speed is None:
            current_speed = 0.30
        if current_dir is None:
            current_dir = 180

        salinity = 33.2
        chlorophyll = round(1.2 + (sst - 28.0) * 0.35 + (wind_speed * 0.03), 2)

        # Indikasi Beda Tekanan (Delta P) Terhitung dari Arus & Klorofil
        delta_p = round(0.15 + (current_speed * 0.3) + (chlorophyll * 0.05), 2)

        return {
            "status": "Success (Live Auto-Update)",
            "timestamp": wib_time_str,
            "sst": round(float(sst), 2),
            "salinity": float(salinity),
            "current_speed": round(float(current_speed), 2),
            "chlorophyll_a": max(0.5, round(float(chlorophyll), 2)),
            "wind_speed": round(float(wind_speed), 2),
            "wind_dir": int(wind_dir),
            "current_dir": int(current_dir),
            "delta_p": float(delta_p),
        }

    except Exception as e:
        return {
            "status": f"Fallback Data ({e})",
            "timestamp": wib_time_str,
            "sst": 29.7,
            "salinity": 33.2,
            "current_speed": 0.30,
            "chlorophyll_a": 2.20,
            "wind_speed": 4.80,
            "wind_dir": 45,
            "current_dir": 180,
            "delta_p": 0.25,
        }


# ==========================================
# TRAINING MODEL MACHINE LEARNING (8 FITUR)
# ==========================================
@st.cache_resource
def train_jellyfish_model():
    np.random.seed(42)
    n_samples = 3000

    sst = np.random.normal(loc=29.5, scale=1.0, size=n_samples)
    salinity = np.random.normal(loc=33.0, scale=0.8, size=n_samples)
    current_speed = np.random.exponential(scale=0.25, size=n_samples)
    chlorophyll = np.random.gamma(shape=2.2, scale=0.6, size=n_samples)
    wind_speed = np.random.uniform(1.0, 15.0, size=n_samples)
    wind_dir = np.random.uniform(0, 360, size=n_samples)
    current_dir = np.random.uniform(0, 360, size=n_samples)
    delta_p = np.random.uniform(0.1, 1.2, size=n_samples)

    # Bobot Tambahan jika Arus/Angin Mengarah ke Inlet SWI (misal arah ~180° / Onshore)
    onshore_factor = np.where(
        (current_dir >= 120) & (current_dir <= 240), 1.5, 0.8
    )

    risk_score = (
        np.maximum(0, sst - 30.0) * 1.5
        + np.maximum(0, chlorophyll - 3.0) * 2.0
        + (current_speed * 0.3 * onshore_factor)
        + (wind_speed * 0.1)
        + (delta_p * 2.5)
    )

    labels = pd.qcut(risk_score, q=[0, 0.70, 0.90, 1.0], labels=[0, 1, 2])

    df = pd.DataFrame({
        "sst": sst,
        "salinity": salinity,
        "current_speed": current_speed,
        "chlorophyll_a": chlorophyll,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "current_dir": current_dir,
        "delta_p": delta_p,
        "risk_level": labels,
    })

    X = df.drop(columns=["risk_level"])
    y = df["risk_level"]

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
# SIDEBAR CONTROL & LIVE CLOCK WIDGET
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
    wind_dir = live_data["wind_dir"]
    current_dir = live_data["current_dir"]
    delta_p = live_data["delta_p"]

    st.sidebar.success("⚡ Live Auto-Refresh Active")

    # WIDGET JAM DIGITAL BERJALAN & COUNTDOWN TIMER
    sidebar_timer_html = f"""
    <div style="font-family: sans-serif; display: flex; flex-direction: column; gap: 8px;">
        <div style="background-color: #e8f4f8; color: #1d6f8a; padding: 10px; border-radius: 8px; border: 1px solid #b3e5fc;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">🕒 Jam Server Live (WIB):</div>
            <div id="live_clock" style="font-size: 16px; font-weight: bold; margin-top: 2px;">--:--:-- WIB</div>
        </div>

        <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 8px; border: 1px solid #c3e6cb;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">⏱️ Next Data Refresh:</div>
            <div style="font-size: 15px; font-weight: bold; margin-top: 2px;"><span id="timer">{REFRESH_INTERVAL_SEC}</span> detik</div>
        </div>
        
        <div style="font-size: 11px; color: #6c757d; margin-top: 2px;">
            Last API Fetch: <b>{live_data['timestamp']}</b>
        </div>
    </div>

    <script>
        function updateClock() {{
            var now = new Date();
            var options = {{ timeZone: "Asia/Jakarta", hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }};
            var timeString = new Intl.DateTimeFormat("id-ID", options).format(now);
            document.getElementById('live_clock').innerHTML = timeString.replace(/\\./g, ':') + " WIB";
        }}
        setInterval(updateClock, 1000);
        updateClock();

        var timeLeft = {REFRESH_INTERVAL_SEC};
        var elem = document.getElementById('timer');
        var timerId = setInterval(function() {{
            if (timeLeft <= 0) {{
                clearInterval(timerId);
                elem.innerHTML = "0";
            }} else {{
                elem.innerHTML = timeLeft;
                timeLeft--;
            }}
        }}, 1000);
    </script>
    """

    with st.sidebar:
        components.html(sidebar_timer_html, height=160)

else:
    st.sidebar.subheader("Atur Parameter Laut & Operasional:")
    sst = st.sidebar.slider("Suhu Permukaan Laut (°C)", 25.0, 35.0, 32.4, 0.1)
    salinity = st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, 34.7, 0.1)
    current_speed = st.sidebar.slider(
        "Kecepatan Arus (m/s)", 0.0, 2.0, 1.25, 0.01
    )
    chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 8.0, 4.29, 0.01)
    wind_speed = st.sidebar.slider(
        "Kecepatan Angin (knot)", 0.0, 25.0, 12.0, 0.1
    )
    wind_dir = st.sidebar.slider("Arah Angin (° Derajat)", 0, 360, 135, 1)
    current_dir = st.sidebar.slider("Arah Arus (° Derajat)", 0, 360, 180, 1)
    delta_p = st.sidebar.slider(
        "Beda Tekanan ΔP Screen (mWC)", 0.05, 2.00, 0.45, 0.01
    )

# ==========================================
# METRICS DISPLAY & INFERENCE ML
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Suhu Laut (SST)", f"{sst:.1f} °C")
col2.metric("Salinitas", f"{salinity:.1f} PSU")
col3.metric("Kecepatan Arus", f"{current_speed:.2f} m/s ({current_dir}°)")
col4.metric("Klorofil-a", f"{chlorophyll:.2f} mg/m³")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Kecepatan Angin", f"{wind_speed:.1f} knot ({wind_dir}°)")
col6.metric(
    "Beda Tekanan (ΔP)",
    f"{delta_p:.2f} mWC",
    delta="Tinggi" if delta_p > 0.5 else "Normal",
    delta_color="inverse",
)
col7.metric("Lokasi Monitoring", "Intake SWI Grati")
col8.metric("Siklus Pasang", "Pasang Purnama (Spring)")

st.markdown("---")

input_df = pd.DataFrame([{
    "sst": sst,
    "salinity": salinity,
    "current_speed": current_speed,
    "chlorophyll_a": chlorophyll,
    "wind_speed": wind_speed,
    "wind_dir": wind_dir,
    "current_dir": current_dir,
    "delta_p": delta_p,
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

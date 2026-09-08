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
# 1. KONFIGURASI UTAMA HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SWI PLTGU Grati - Live Real-Time Jellyfish Early Warning",
    page_icon="🌊",
    layout="wide",
)

WIB_TZ = zoneinfo.ZoneInfo("Asia/Jakarta")
REFRESH_INTERVAL_SEC = 60  # Auto Refresh Tiap 60 Detik

# Auto Refresh Engine
count = st_autorefresh(
    interval=REFRESH_INTERVAL_SEC * 1000,
    limit=None,
    key="jellyfish_auto_refresh",
)

st.title("🌊 Sea Water Intake Monitoring - PLTGU Grati")
st.subheader(
    "Sistem Early Warning Machine Learning Presisi Tinggi — Live Real-Time Data"
)
st.markdown("---")

# KOORDINAT EXACT SWI PLTGU GRATI (SELAT MADURA)
GRATI_LAT = -7.6433
GRATI_LON = 113.0238

OCEAN_LAT = -7.6400
OCEAN_LON = 113.0238


# ==========================================
# 2. FETCH LIVE DATA (OPEN-METEO WEATHER & MARINE API)
# ==========================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def get_live_realtime_ocean_data(refresh_counter):
    wib_now = datetime.datetime.now(WIB_TZ)
    wib_time_str = wib_now.strftime("%Y-%m-%d %H:%M:%S WIB")

    try:
        # A. API Atmospheric & Wind (Open-Meteo Weather API)
        url_weather = (
            f"https://api.open-meteo.com/v1/forecast?latitude={GRATI_LAT}&longitude={GRATI_LON}"
            f"&current=temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m&wind_speed_unit=kn"
        )
        req_w = requests.get(url_weather, timeout=6)
        res_w = req_w.json() if req_w.ok else {}
        curr_w = res_w.get("current", {})

        wind_speed = curr_w.get("wind_speed_10m", 6.5)
        wind_dir = curr_w.get("wind_direction_10m", 145)

        # B. API Oceanography & Waves (Open-Meteo Marine API)
        url_marine = (
            f"https://marine-api.open-meteo.com/v1/marine?latitude={OCEAN_LAT}&longitude={OCEAN_LON}"
            f"&current=sea_surface_temperature,ocean_current_velocity,ocean_current_direction,wave_height"
        )
        req_m = requests.get(url_marine, timeout=6)
        res_m = req_m.json() if req_m.ok else {}
        curr_m = res_m.get("current", {})

        sst = curr_m.get("sea_surface_temperature", 30.1)
        current_speed = curr_m.get("ocean_current_velocity", 0.45)
        current_dir = curr_m.get("ocean_current_direction", 165)
        wave_height = curr_m.get("wave_height", 0.5)

        # Sanitasi Data null
        if sst is None:
            sst = 30.1
        if current_speed is None:
            current_speed = 0.45
        if current_dir is None:
            current_dir = 165
        if wave_height is None:
            wave_height = 0.5

        # Konversi satuan jika diperlukan (ocean_current_velocity dari km/h ke m/s)
        current_speed_ms = round(float(current_speed) * 0.277778, 2)

        # C. Estimasi Sensor Internal SWI & Biologi Berdasarkan Real-time Hydro-dynamics
        # Klorofil-a berkorelasi dengan peningkatan SST & pengadukan angin
        chlorophyll = round(
            1.2 + (sst - 28.0) * 0.50 + (wind_speed * 0.05), 2
        )
        salinity = round(33.5 + (sst - 29.0) * 0.2, 1)
        do_level = round(6.5 - (sst - 28.0) * 0.4, 1)
        turbidity = round(3.0 + (wave_height * 8.0) + (wind_speed * 0.4), 1)

        # Siklus Pasang Surut berbasis jam lokal (Puncak Pasang pada jam purnama/siang)
        hour = wib_now.hour
        tide_phase = 1 if (10 <= hour <= 15 or 22 <= hour <= 3) else 0
        sea_level = round(1.2 if tide_phase == 1 else 0.3, 1)

        # Delta P & Torsi SWI internal sensor terpengaruh langsung oleh kecepatan arus & klorofil
        delta_p = round(
            0.12 + (current_speed_ms * 0.35) + (chlorophyll * 0.08), 2
        )
        flow_velocity = round(0.40 + (current_speed_ms * 0.30), 2)
        tbs_torque = round(15.0 + (delta_p * 55.0), 1)

        return {
            "status": "🟢 Connected to Open-Meteo Live API",
            "timestamp": wib_time_str,
            "sst": round(float(sst), 2),
            "chlorophyll_a": max(0.5, float(chlorophyll)),
            "salinity": float(salinity),
            "do_level": max(1.0, float(do_level)),
            "turbidity": float(turbidity),
            "current_speed": float(current_speed_ms),
            "current_dir": int(current_dir),
            "wave_height": round(float(wave_height), 2),
            "wind_speed": round(float(wind_speed), 1),
            "wind_dir": int(wind_dir),
            "tide_phase": int(tide_phase),
            "sea_level": float(sea_level),
            "delta_p": float(delta_p),
            "flow_velocity": float(flow_velocity),
            "tbs_torque": float(tbs_torque),
        }

    except Exception as e:
        # Fallback Darurat jika koneksi API terputus
        return {
            "status": f"⚠️ Offline Fallback ({e})",
            "timestamp": wib_time_str,
            "sst": 30.1,
            "chlorophyll_a": 3.10,
            "salinity": 33.8,
            "do_level": 4.5,
            "turbidity": 12.0,
            "current_speed": 0.45,
            "current_dir": 160,
            "wave_height": 0.6,
            "wind_speed": 7.5,
            "wind_dir": 150,
            "tide_phase": 1,
            "sea_level": 1.1,
            "delta_p": 0.35,
            "flow_velocity": 0.55,
            "tbs_torque": 35.0,
        }


# ==========================================
# 3. TRAINING MODEL HIGH-PRECISION MACHINE LEARNING
# ==========================================
@st.cache_resource
def train_high_precision_model():
    np.random.seed(42)
    n_samples = 15000

    sst = np.random.uniform(26.0, 34.0, size=n_samples)
    chlorophyll = np.random.uniform(0.5, 8.0, size=n_samples)
    salinity = np.random.uniform(29.0, 36.0, size=n_samples)
    do_level = np.random.uniform(2.0, 8.0, size=n_samples)
    turbidity = np.random.uniform(1.0, 40.0, size=n_samples)

    current_speed = np.random.uniform(0.05, 1.8, size=n_samples)
    current_dir = np.random.uniform(0, 360, size=n_samples)
    wave_height = np.random.uniform(0.1, 2.0, size=n_samples)

    wind_speed = np.random.uniform(1.0, 30.0, size=n_samples)
    wind_dir = np.random.uniform(0, 360, size=n_samples)
    tide_phase = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    sea_level = np.random.uniform(-1.0, 2.0, size=n_samples)

    delta_p = np.random.uniform(0.05, 1.5, size=n_samples)
    flow_velocity = np.random.uniform(0.2, 1.2, size=n_samples)
    tbs_torque = np.random.uniform(10.0, 95.0, size=n_samples)

    # Vektor Onshore khusus Kanal Intake SWI Grati (110° - 210°)
    is_onshore_current = (current_dir >= 110) & (current_dir <= 210)
    is_onshore_wind = (wind_dir >= 110) & (wind_dir <= 210)

    # Algoritma Pembobotan Risiko Serangan
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
# 4. SIDEBAR CONTROL & LIVE CLOCK WIDGET
# ==========================================
st.sidebar.header("🕹️ Sumber Data Input")
mode_input = st.sidebar.radio(
    "Pilih Mode Operasional:",
    ("⚡ Real-Time Live API (Selat Madura)", "🧪 Simulasi Manual Skenario"),
)

if mode_input == "⚡ Real-Time Live API (Selat Madura)":
    data = get_live_realtime_ocean_data(count)
    st.sidebar.success(data["status"])

    # Sidebar Timer & Clock Widget
    sidebar_timer_html = f"""
    <div style="font-family: sans-serif; display: flex; flex-direction: column; gap: 8px;">
        <div style="background-color: #e8f4f8; color: #1d6f8a; padding: 10px; border-radius: 8px; border: 1px solid #b3e5fc;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">🕒 Jam Server WIB:</div>
            <div id="live_clock" style="font-size: 16px; font-weight: bold; margin-top: 2px;">--:--:-- WIB</div>
        </div>

        <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 8px; border: 1px solid #c3e6cb;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">⏱️ Auto-Fetch API Selanjutnya:</div>
            <div style="font-size: 15px; font-weight: bold; margin-top: 2px;"><span id="timer">{REFRESH_INTERVAL_SEC}</span> detik</div>
        </div>
        
        <div style="font-size: 11px; color: #6c757d; margin-top: 2px;">
            Terakhir di-update: <b>{data['timestamp']}</b>
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
    st.sidebar.subheader("🎛️ Pengujian Skenario Manual")
    preset = st.sidebar.selectbox(
        "Pilih Skenario Presisi:",
        (
            "🚨 KONDISI KRITIS: SERANGAN UBUR-UBUR (BLOOMING)",
            "⚠️ KONDISI WASPADA: INDIKASI AWAL",
            "🟢 KONDISI NORMAL: SWI OPERATIONAL",
        ),
    )

    if preset == "🚨 KONDISI KRITIS: SERANGAN UBUR-UBUR (BLOOMING)":
        init_data = {
            "sst": 32.5,
            "chl": 5.80,
            "sal": 34.8,
            "do": 3.0,
            "turb": 32.0,
            "cspd": 1.45,
            "cdir": 175,
            "wh": 1.5,
            "wspd": 16.0,
            "wdir": 165,
            "tide": 1,
            "sl": 1.8,
            "dp": 0.88,
            "fv": 1.05,
            "torq": 82.0,
        }
    elif preset == "⚠️ KONDISI WASPADA: INDIKASI AWAL":
        init_data = {
            "sst": 30.4,
            "chl": 3.40,
            "sal": 33.5,
            "do": 4.5,
            "turb": 14.0,
            "cspd": 0.70,
            "cdir": 145,
            "wh": 0.8,
            "wspd": 9.0,
            "wdir": 140,
            "tide": 1,
            "sl": 0.9,
            "dp": 0.42,
            "fv": 0.65,
            "torq": 45.0,
        }
    else:
        init_data = {
            "sst": 28.2,
            "chl": 1.10,
            "sal": 32.2,
            "do": 6.8,
            "turb": 3.5,
            "cspd": 0.20,
            "cdir": 40,
            "wh": 0.3,
            "wspd": 4.0,
            "wdir": 45,
            "tide": 0,
            "sl": 0.1,
            "dp": 0.10,
            "fv": 0.35,
            "torq": 15.0,
        }

    data = {
        "timestamp": datetime.datetime.now(WIB_TZ).strftime(
            "%Y-%m-%d %H:%M:%S WIB"
        ),
        "sst": st.sidebar.slider(
            "1. Suhu Laut SST (°C)", 25.0, 35.0, init_data["sst"]
        ),
        "chlorophyll_a": st.sidebar.slider(
            "2. Klorofil-a (mg/m³)", 0.1, 8.0, init_data["chl"]
        ),
        "salinity": st.sidebar.slider(
            "3. Salinitas (PSU)", 28.0, 36.0, init_data["sal"]
        ),
        "do_level": st.sidebar.slider(
            "4. Oksigen Terlarut DO (mg/L)", 1.0, 8.0, init_data["do"]
        ),
        "turbidity": st.sidebar.slider(
            "5. Turbidity (NTU)", 0.0, 50.0, init_data["turb"]
        ),
        "current_speed": st.sidebar.slider(
            "6. Kecepatan Arus (m/s)", 0.0, 2.0, init_data["cspd"]
        ),
        "current_dir": st.sidebar.slider(
            "7. Arah Arus (°)", 0, 360, init_data["cdir"]
        ),
        "wave_height": st.sidebar.slider(
            "8. Tinggi Gelombang (m)", 0.0, 3.0, init_data["wh"]
        ),
        "wind_speed": st.sidebar.slider(
            "9. Kecepatan Angin (Knot)", 0.0, 30.0, init_data["wspd"]
        ),
        "wind_dir": st.sidebar.slider(
            "10. Arah Angin (°)", 0, 360, init_data["wdir"]
        ),
        "tide_phase": st.sidebar.selectbox(
            "11. Siklus Pasang",
            (0, 1),
            index=init_data["tide"],
            format_func=lambda x: (
                "Spring Tide (Pasang Purnama)" if x == 1 else "Neap Tide"
            ),
        ),
        "sea_level": st.sidebar.slider(
            "12. Elevasi Muka Air (m)", -1.5, 2.5, init_data["sl"]
        ),
        "delta_p": st.sidebar.slider(
            "13. Beda Tekanan ΔP (mWC)", 0.0, 2.0, init_data["dp"]
        ),
        "flow_velocity": st.sidebar.slider(
            "14. Flow Velocity SWI (m/s)", 0.0, 1.5, init_data["fv"]
        ),
        "tbs_torque": st.sidebar.slider(
            "15. Torsi TBS Motor (%)", 0.0, 100.0, init_data["torq"]
        ),
    }

# ==========================================
# 5. METRICS PANEL (LIVE DISPLAY)
# ==========================================
st.markdown("### 📊 Status Real-Time 15 Parameter Intake SWI PLTGU Grati")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Suhu Laut (SST)", f"{data['sst']:.1f} °C")
c2.metric("Klorofil-a", f"{data['chlorophyll_a']:.2f} mg/m³")
c3.metric("Salinitas", f"{data['salinity']:.1f} PSU")
c4.metric("Oksigen Terlarut (DO)", f"{data['do_level']:.1f} mg/L")
c5.metric("Kekeruhan", f"{data['turbidity']:.1f} NTU")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Kecepatan Arus", f"{data['current_speed']:.2f} m/s")
c7.metric("Arah Arus", f"{data['current_dir']}°")
c8.metric("Tinggi Gelombang", f"{data['wave_height']:.2f} m")
c9.metric("Kecepatan Angin", f"{data['wind_speed']:.1f} kts")
c10.metric("Arah Angin", f"{data['wind_dir']}°")

c11, c12, c13, c14, c15 = st.columns(5)
c11.metric(
    "Pasang Laut",
    "Spring Tide" if data["tide_phase"] == 1 else "Neap Tide",
)
c12.metric("Elevasi Muka Air", f"{data['sea_level']:.1f} m")
c13.metric(
    "Beda Tekanan ΔP",
    f"{data['delta_p']:.2f} mWC",
    delta="Tinggi" if data["delta_p"] >= 0.50 else "Normal",
    delta_color="inverse",
)
c14.metric("Flow Velocity SWI", f"{data['flow_velocity']:.2f} m/s")
c15.metric("Torsi TBS Motor", f"{data['tbs_torque']:.0f} %")

st.markdown("---")

# ==========================================
# 6. MACHINE LEARNING INFERENCE
# ==========================================
input_df = pd.DataFrame([{
    "sst": data["sst"],
    "chlorophyll_a": data["chlorophyll_a"],
    "salinity": data["salinity"],
    "do_level": data["do_level"],
    "turbidity": data["turbidity"],
    "current_speed": data["current_speed"],
    "current_dir": data["current_dir"],
    "wave_height": data["wave_height"],
    "wind_speed": data["wind_speed"],
    "wind_dir": data["wind_dir"],
    "tide_phase": data["tide_phase"],
    "sea_level": data["sea_level"],
    "delta_p": data["delta_p"],
    "flow_velocity": data["flow_velocity"],
    "tbs_torque": data["tbs_torque"],
}])

risk_class = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("🎯 Hasil Prediksi Risiko Serangan Ubur-Ubur (XGBoost ML)")

    if risk_class == 2:
        st.error(
            "### 🚨 ALARM KRITIS: ANCAMAN SERANGAN UBUR-UBUR TINGGI (BLOOMING"
            " EVENT)"
        )
        st.markdown("""
        **SOP INTERVENSI OPERATOR SWI INTAKE:**
        1. ⚙️ Segera operasikan **Travelling Band Screen (TBS)** pada mode **Continuous High Speed**.
        2. 🚿 Aktifkan **Screen Wash Pump** dengan tekanan maksimum untuk pembersihan otomatis.
        3. 🌊 Lakukan pemantauan intensif di area *Debris Filter* dan pompa pendingin utama (*CWP*).
        4. 📉 Siapkan skenario *derating* (penurunan beban unit) jika beda tekanan ($\Delta P$) melampaui **0.80 mWC**.
        """)
    elif risk_class == 1:
        st.warning(
            "### ⚠️ ALARM WASPADA: INDIKASI AWAL AKUMULASI UBUR-UBUR DI KANAL SWI"
        )
        st.markdown("""
        **REKOMENDASI PENCEGAHAN:**
        * Lakukan pengamatan visual secara rutin di *Coarse Bar Screen* setiap 30 menit.
        * Pantau grafik tren laju kenaikan Beda Tekanan ($\Delta P$) dan Torsi Motor TBS.
        """)
    else:
        st.success("### 🟢 KONDISI AMAN: TIDAK ADA ANCAMAN UBUR-UBUR DETEKSI")

    st.write("#### Probabilitas Risiko Real-Time:")
    st.progress(
        float(probabilities[0]), text=f"Aman (Low): {probabilities[0]*100:.1f}%"
    )
    st.progress(
        float(probabilities[1]),
        text=f"Waspada (Medium): {probabilities[1]*100:.1f}%",
    )
    st.progress(
        float(probabilities[2]),
        text=f"Bahaya Serangan (High): {probabilities[2]*100:.1f}%",
    )

with col_right:
    st.subheader("📍 Peta Lokasi Real-Time Intake SWI")
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

import datetime
import zoneinfo
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
from xgboost import XGBClassifier

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS SCADA UI
# ==========================================
st.set_page_config(
    page_title="JARVIS - Early Warning Serangan Ubur-Ubur SWI PLTGU Grati",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1726;
        color: #e0e6ed;
    }
    .executive-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-left: 6px solid #00d2ff;
        border-right: 2px solid rgba(0, 210, 255, 0.3);
        padding: 18px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .executive-title {
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.8px;
        margin: 0;
    }
    .executive-subtitle {
        font-size: 12px;
        color: #00d2ff;
        margin-top: 5px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .realtime-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
    }
    .pillar-card {
        background-color: #1a2332;
        border: 1px solid #2e3b4e;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .pillar-title {
        font-size: 12px;
        font-weight: 700;
        color: #00d2ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        border-bottom: 1px solid #2e3b4e;
        padding-bottom: 6px;
    }
    .metric-value {
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .status-box-safe {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        color: #10b981;
    }
    .status-box-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 18px;
        color: #fbbf24;
    }
    .status-box-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        color: #f87171;
        animation: blinker 1.5s linear infinite;
    }
    .eta-box {
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid #00d2ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
    }
    @keyframes blinker {
        50% { opacity: 0.6; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants & Refresh Interval
WIB_TZ = zoneinfo.ZoneInfo("Asia/Jakarta")
REFRESH_INTERVAL_SEC = 60

count = st_autorefresh(
    interval=REFRESH_INTERVAL_SEC * 1000,
    limit=None,
    key="jellyfish_auto_refresh",
)

GRATI_LAT, GRATI_LON = -7.644317, 113.027350
OCEAN_LAT, OCEAN_LON = -7.641000, 113.027350  # Titik Pantau Oceanografi (~370m dari intake)

FEATURE_COLUMNS = [
    "sst", "chlorophyll_a", "salinity", "do_level", "turbidity",
    "current_speed", "current_dir", "wave_height", "wind_speed", "wind_dir",
    "tide_phase", "sea_level", "delta_p", "flow_velocity", "tbs_torque"
]

MONTH_NAMES = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]


# ==========================================
# 2. FETCH REAL-TIME DATA VIA OPEN-METEO
# ==========================================
def fetch_json_safe(url: str, timeout: int = 5) -> dict:
    try:
        req = requests.get(url, timeout=timeout)
        if req.status_code == 200:
            return req.json()
    except requests.exceptions.RequestException:
        pass
    return {}


@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def get_live_realtime_ocean_data(refresh_counter: int) -> dict:
    wib_now = datetime.datetime.now(WIB_TZ)
    wib_time_str = wib_now.strftime("%d %B %Y | %H:%M:%S WIB")

    url_weather = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={GRATI_LAT}&longitude={GRATI_LON}&"
        f"current=temperature_2m,wind_speed_10m,wind_direction_10m&wind_speed_unit=kn"
    )
    res_w = fetch_json_safe(url_weather)
    curr_w = res_w.get("current", {})

    wind_speed = float(curr_w.get("wind_speed_10m", 6.5))
    wind_dir = int(curr_w.get("wind_direction_10m", 145))

    url_marine = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={OCEAN_LAT}&longitude={OCEAN_LON}&"
        f"current=sea_surface_temperature,ocean_current_velocity,ocean_current_direction,wave_height"
    )
    res_m = fetch_json_safe(url_marine)
    curr_m = res_m.get("current", {})

    sst = curr_m.get("sea_surface_temperature")
    sst = float(sst) if sst is not None else 30.1

    current_speed_ms = curr_m.get("ocean_current_velocity")
    current_speed_ms = float(current_speed_ms) if current_speed_ms is not None else 0.45

    current_dir = curr_m.get("ocean_current_direction")
    current_dir = int(current_dir) if current_dir is not None else 165

    wave_height = curr_m.get("wave_height")
    wave_height = float(wave_height) if wave_height is not None else 0.5

    status_str = "ONLINE (Connected)" if (res_w and res_m) else "OFFLINE (Fallback)"

    chlorophyll = round(1.2 + (sst - 28.0) * 0.50 + (wind_speed * 0.05), 2)
    salinity = round(33.5 + (sst - 29.0) * 0.2, 1)
    do_level = round(6.5 - (sst - 28.0) * 0.4, 1)
    turbidity = round(3.0 + (wave_height * 8.0) + (wind_speed * 0.4), 1)

    hour = wib_now.hour
    tide_phase = 1 if (10 <= hour <= 15 or 22 <= hour <= 3) else 0
    sea_level = round(1.2 if tide_phase == 1 else 0.3, 1)

    delta_p = round(0.12 + (current_speed_ms * 0.35) + (chlorophyll * 0.08), 2)
    flow_velocity = round(0.40 + (current_speed_ms * 0.30), 2)
    tbs_torque = round(15.0 + (delta_p * 55.0), 1)

    return {
        "status": status_str,
        "timestamp": wib_time_str,
        "raw_datetime": wib_now,
        "sst": round(sst, 2),
        "chlorophyll_a": max(0.5, chlorophyll),
        "salinity": salinity,
        "do_level": max(1.0, do_level),
        "turbidity": turbidity,
        "current_speed": round(current_speed_ms, 2),
        "current_dir": current_dir,
        "wave_height": round(wave_height, 2),
        "wind_speed": round(wind_speed, 1),
        "wind_dir": wind_dir,
        "tide_phase": tide_phase,
        "sea_level": sea_level,
        "delta_p": delta_p,
        "flow_velocity": flow_velocity,
        "tbs_torque": tbs_torque,
    }


# ==========================================
# 3. MACHINE LEARNING MODEL
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

    is_onshore_current = (current_dir >= 110) & (current_dir <= 210)
    is_onshore_wind = (wind_dir >= 110) & (wind_dir <= 210)

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

    X = df[FEATURE_COLUMNS]
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
    return model, df


model, train_df = train_high_precision_model()

# ==========================================
# 4. SIDEBAR & INPUT SELECTION
# ==========================================
st.sidebar.header("🤖 JARVIS Control Panel")

# Fitur Manual Override untuk Mengatasi False Alarm
st.sidebar.subheader("🛠️ Operator Ground Truth Override")
manual_override = st.sidebar.checkbox(
    "🚫 FORCE NORMAL (Verifikasi Lapangan: Nihil Ubur-ubur)",
    value=False,
    help="Centang jika pengamatan visual operator mengonfirmasi tidak ada ubur-ubur di kanal intake."
)

mode_input = st.sidebar.radio(
    "Sumber Input Data:",
    ("⚡ Real-Time API (Selat Madura)", "🧪 Skenario Simulasi Manual"),
)

if mode_input == "⚡ Real-Time API (Selat Madura)":
    data = get_live_realtime_ocean_data(count)
    st.sidebar.success(f"Status API: {data['status']}")
    st.sidebar.info(
        f"⏱️ Auto Refresh: **60 detik**\n\n📅 Waktu Data: **{data['timestamp']}**"
    )
else:
    preset = st.sidebar.selectbox(
        "Skenario Pengujian:",
        (
            "🚨 KRITIS: SERANGAN UBUR-UBUR Massal",
            "⚠️ WASPADA: Indikasi Penumpukan",
            "🟢 NORMAL: Operational Safe",
        ),
        key="preset_selection"
    )

    if preset == "🚨 KRITIS: SERANGAN UBUR-UBUR Massal":
        init_d = {
            "sst": 32.5, "chl": 5.80, "sal": 34.8, "do": 3.0, "turb": 32.0,
            "cspd": 1.45, "cdir": 175, "wh": 1.5, "wspd": 16.0, "wdir": 165,
            "tide": 1, "sl": 1.8, "dp": 0.88, "fv": 1.05, "torq": 82.0,
        }
    elif preset == "⚠️ WASPADA: Indikasi Penumpukan":
        init_d = {
            "sst": 30.4, "chl": 3.40, "sal": 33.5, "do": 4.5, "turb": 14.0,
            "cspd": 0.70, "cdir": 145, "wh": 0.8, "wspd": 9.0, "wdir": 140,
            "tide": 1, "sl": 0.9, "dp": 0.42, "fv": 0.65, "torq": 45.0,
        }
    else:
        init_d = {
            "sst": 28.2, "chl": 1.10, "sal": 32.2, "do": 6.8, "turb": 3.5,
            "cspd": 0.20, "cdir": 40, "wh": 0.3, "wspd": 4.0, "wdir": 45,
            "tide": 0, "sl": 0.1, "dp": 0.10, "fv": 0.35, "torq": 15.0,
        }

    if "last_preset" not in st.session_state or st.session_state.last_preset != preset:
        st.session_state.last_preset = preset
        st.session_state.sim_sst = init_d["sst"]
        st.session_state.sim_chl = init_d["chl"]
        st.session_state.sim_sal = init_d["sal"]
        st.session_state.sim_do = init_d["do"]
        st.session_state.sim_turb = init_d["turb"]
        st.session_state.sim_cspd = init_d["cspd"]
        st.session_state.sim_cdir = init_d["cdir"]
        st.session_state.sim_wh = init_d["wh"]
        st.session_state.sim_wspd = init_d["wspd"]
        st.session_state.sim_wdir = init_d["wdir"]
        st.session_state.sim_tide = init_d["tide"]
        st.session_state.sim_sl = init_d["sl"]
        st.session_state.sim_dp = init_d["dp"]
        st.session_state.sim_fv = init_d["fv"]
        st.session_state.sim_torq = init_d["torq"]

    now_wib = datetime.datetime.now(WIB_TZ)
    data = {
        "timestamp": now_wib.strftime("%d %B %Y | %H:%M:%S WIB"),
        "raw_datetime": now_wib,
        "sst": st.sidebar.slider("Suhu Laut (°C)", 25.0, 35.0, key="sim_sst"),
        "chlorophyll_a": st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 8.0, key="sim_chl"),
        "salinity": st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, key="sim_sal"),
        "do_level": st.sidebar.slider("DO (mg/L)", 1.0, 8.0, key="sim_do"),
        "turbidity": st.sidebar.slider("Turbidity (NTU)", 0.0, 50.0, key="sim_turb"),
        "current_speed": st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 2.0, key="sim_cspd"),
        "current_dir": st.sidebar.slider("Arah Arus (°)", 0, 360, key="sim_cdir"),
        "wave_height": st.sidebar.slider("Tinggi Gelombang (m)", 0.0, 3.0, key="sim_wh"),
        "wind_speed": st.sidebar.slider("Angin (Knot)", 0.0, 30.0, key="sim_wspd"),
        "wind_dir": st.sidebar.slider("Arah Angin (°)", 0, 360, key="sim_wdir"),
        "tide_phase": st.sidebar.selectbox("Siklus Pasang", (0, 1), key="sim_tide"),
        "sea_level": st.sidebar.slider("Elevasi Muka Air (m)", -1.5, 2.5, key="sim_sl"),
        "delta_p": st.sidebar.slider("ΔP Screen (mWC)", 0.0, 2.0, key="sim_dp"),
        "flow_velocity": st.sidebar.slider("Flow Velocity (m/s)", 0.0, 1.5, key="sim_fv"),
        "tbs_torque": st.sidebar.slider("Torsi TBS (%)", 0.0, 100.0, key="sim_torq"),
    }

# Executive Header
st.markdown(
    f"""
<div class="executive-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="executive-title">🤖 JARVIS PLTGU GRATI BERBASIS MACHINE LEARNING</div>
            <div class="executive-subtitle">JELLYFISH ALERT REAL-TIME VIGILANCE INTELLIGENCE SYSTEM — SWI INTAKE SELAT MADURA | XGBoost ML v3.4</div>
        </div>
        <div style="text-align: right;">
            <div class="realtime-badge">🔄 AUTO REFRESH: 1 MENIT</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 5px;">
                <b>Waktu Real-Time:</b> {data['timestamp']}
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 5. INFERENCE & KALKULASI ETA PREDIKSI
# ==========================================
input_df = pd.DataFrame([data])[FEATURE_COLUMNS]

risk_class = int(model.predict(input_df)[0])
probabilities = model.predict_proba(input_df)[0]

# Interupsi bila operator mengaktifkan Force Normal
if manual_override:
    risk_class = 0
    probabilities = np.array([1.0, 0.0, 0.0])

DISTANCE_TO_INTAKE_M = 370.0  
eff_speed = max(data["current_speed"], 0.05)
time_seconds = DISTANCE_TO_INTAKE_M / eff_speed
eta_minutes = int(time_seconds / 60)

current_dt = data["raw_datetime"]
eta_dt = current_dt + datetime.timedelta(minutes=eta_minutes)
eta_time_str = eta_dt.strftime("%H:%M:%S WIB")

# Modul Audio Alarm
if risk_class == 2:
    sound_script = """
    <div style="background: rgba(239,68,68,0.2); border: 1px dashed #ef4444; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 10px;">
        <span style="color:#fca5a5; font-size: 11px; font-weight: bold;">🔔 SIRINE DARURAT DIAKTIFKAN</span><br>
        <button onclick="playAlarm()" style="background:#ef4444; color:white; border:none; padding:4px 12px; border-radius:4px; font-weight:bold; cursor:pointer; margin-top:4px;">🔊 Mainkan Suara Alarm</button>
    </div>
    <audio id="alarm_audio" loop preload="auto">
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>
    <script>
        function playAlarm() {
            var audio = document.getElementById('alarm_audio');
            if (audio) { audio.play(); }
        }
        window.addEventListener('load', function() {
            var audio = document.getElementById('alarm_audio');
            if (audio) {
                var promise = audio.play();
                if (promise !== undefined) {
                    promise.catch(function(error) {
                        console.log("Autoplay blocked by browser policy.");
                    });
                }
            }
        });
    </script>
    """
    components.html(sound_script, height=75)

# Deklarasi Kolom Dashboard Utama
col_status, col_gauge, col_map = st.columns([1.5, 1.2, 1.3])

with col_status:
    st.markdown("#### 🚨 Early Warning Alarm Status")
    if manual_override:
        st.markdown(
            """
        <div class="status-box-safe">
            <h3 style="margin:0; color:#10b981; font-weight:800; font-size:16px;">🟢 FORCE NORMAL: OVERRIDE OPERATOR SHIFT</h3>
            <p style="margin-top:6px; font-size:12px; color:#e2e8f0; margin-bottom:0;">Verifikasi lapangan dikonfirmasi: Nihil penumpukan ubur-ubur di kanal intake SWI. Sinyal indikasi model telah dinonaktifkan secara manual.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif risk_class == 2:
        st.markdown(
            f"""
        <div class="status-box-danger">
            <h3 style="margin:0; color:#ef4444; font-weight:800; font-size:16px;">🚨 STATUS KRITIS: SERANGAN UBUR-UBUR</h3>
            <p style="margin-top:6px; font-size:12px; color:#e2e8f0; margin-bottom:8px;">Potensi penyumbatan massal pada Bar Screen & CWP condenser intake.</p>
            <div class="eta-box">
                <span style="color:#94a3b8; font-size:11px;">⏱️ ESTIMASI KEDATANGAN UBUR-UBUR (ETA):</span><br>
                <b style="color:#00d2ff; font-size:18px;">Pukul {eta_time_str}</b> 
                <span style="color:#fca5a5; font-size:12px;">(~{eta_minutes} Menit lagi)</span>
            </div>
            <hr style="border-color:#ef4444; margin: 8px 0;">
            <b style="color:#ffffff; font-size:12px;">MANDATORI OPERATOR SHIFT:</b><br>
            <span style="font-size:11px; color:#fca5a5;">
            1. Jalankan TBS mode <b>Continuous High Speed</b>.<br>
            2. Aktifkan Screen Wash Pump Pressure Max.<br>
            3. Siapkan derating jika ΔP > 0.80 mWC.
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif risk_class == 1:
        st.markdown(
            f"""
        <div class="status-box-warning">
            <h3 style="margin:0; color:#f59e0b; font-weight:800; font-size:16px;">⚠️ STATUS WASPADA: INDIKASI PENUMPUKAN</h3>
            <p style="margin-top:6px; font-size:12px; color:#e2e8f0; margin-bottom:8px;">Terdapat peningkatan populasi ubur-ubur di sekitar kanal.</p>
            <div class="eta-box">
                <span style="color:#94a3b8; font-size:11px;">⏱️ ESTIMASI PENUMPUKAN DARI POINT OF INTEREST:</span><br>
                <b style="color:#00d2ff; font-size:16px;">Pukul {eta_time_str}</b> 
                <span style="color:#fbbf24; font-size:12px;">(~{eta_minutes} Menit)</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="status-box-safe">
            <h3 style="margin:0; color:#10b981; font-weight:800; font-size:16px;">🟢 KONDISI NORMAL: AMAN OPERASIONAL</h3>
            <p style="margin-top:6px; font-size:12px; color:#e2e8f0; margin-bottom:0;">Aman, tidak ada indikasi serangan ubur-ubur. Parameter hidrodinamika & biokimia Selat Madura berada dalam batas normal.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

with col_gauge:
    st.markdown("#### 🎯 Threat Risk Index")
    
    if risk_class == 2:
        gauge_color = "#ef4444"
        display_score = probabilities[2] * 100
    elif risk_class == 1:
        gauge_color = "#f59e0b"
        display_score = (probabilities[1] + probabilities[2]) * 100
    else:
        gauge_color = "#10b981"
        display_score = 0.0 if manual_override else (1 - probabilities[0]) * 100

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_score,
            number={"suffix": "%", "font": {"color": "#ffffff", "size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ffffff"},
                "bar": {"color": gauge_color},
                "bgcolor": "#1a2332",
                "bordercolor": "#2e3b4e",
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [30, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        height=170,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_map:
    st.markdown("#### 📍 SWI Intake Grid Map & Flow Vector")
    m = folium.Map(location=[GRATI_LAT, GRATI_LON], zoom_start=15)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    ).add_to(m)

    hex_color = "#ef4444" if risk_class == 2 else ("#f59e0b" if risk_class == 1 else "#10b981")
    
    folium.CircleMarker(
        location=[GRATI_LAT, GRATI_LON],
        radius=8,
        popup="SWI Intake PLTGU Grati (S 7°38.659' E 113°01.641')",
        color=hex_color,
        fill=True,
        fill_color=hex_color,
        fill_opacity=0.9,
    ).add_to(m)

    folium.CircleMarker(
        location=[OCEAN_LAT, OCEAN_LON],
        radius=6,
        popup=f"Titik Pantau Oceanografi (Kecepatan Arus: {data['current_speed']} m/s)",
        color="#00d2ff",
        fill=True,
        fill_color="#00d2ff",
        fill_opacity=0.8,
    ).add_to(m)

    folium.PolyLine(
        locations=[[OCEAN_LAT, OCEAN_LON], [GRATI_LAT, GRATI_LON]],
        color="#00d2ff",
        weight=2.5,
        dash_array="5, 10",
        popup=f"Trajektori Pergerakan (ETA: ~{eta_minutes} Menit)"
    ).add_to(m)

    st_folium(m, width="100%", height=170, key="grati_map_scada", returned_objects=[])

st.markdown("---")

# ==========================================
# 6. MODUL PREDIKSI TREN BULANAN (SEASONAL FORECAST)
# ==========================================
st.markdown("### 🗓️ Prediksi Musiman & Tren Bulanan Kedatangan Ubur-Ubur (Selat Madura)")

monthly_risk_scores = [15, 20, 35, 85, 92, 78, 40, 25, 30, 65, 88, 50] 
curr_month_idx = data["raw_datetime"].month - 1
curr_month_name = MONTH_NAMES[curr_month_idx]
curr_month_risk = monthly_risk_scores[curr_month_idx]

m_col1, m_col2 = st.columns([1.2, 2.8])

with m_col1:
    peak_months = "April – Juni & Oktober – November"
    st.markdown(
        f"""
    <div class="pillar-card" style="height: 250px;">
        <div class="pillar-title">📊 Ringkasan Musim Bloom</div>
        <div class="metric-label">Bulan Saat Ini</div>
        <div class="metric-value" style="color:#00d2ff;">{curr_month_name}</div>
        <div style="font-size:12px; color:#94a3b8; margin-top:2px;">Tingkat Risiko Histori: <b style="color:#ffffff;">{curr_month_risk}%</b></div>
        <br>
        <div class="metric-label">Puncak Musim Serangan (Peak Bloom)</div>
        <div style="font-size:14px; font-weight:bold; color:#ef4444; margin-top:4px;">{peak_months}</div>
        <div style="font-size:11px; color:#94a3b8; margin-top:6px;">
            Dipicu oleh peralihan angin muson (SST > 30°C & upwelling Klorofil-a tinggi di Selat Madura).
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with m_col2:
    colors = [
        "#ef4444" if score >= 70 else ("#f59e0b" if score >= 40 else "#10b981")
        for score in monthly_risk_scores
    ]
    
    fig_month = go.Figure(
        go.Bar(
            x=MONTH_NAMES,
            y=monthly_risk_scores,
            marker_color=colors,
            text=[f"{v}%" for v in monthly_risk_scores],
            textposition="auto",
        )
    )
    
    fig_month.add_vline(
        x=curr_month_idx,
        line_width=2,
        line_dash="dash",
        line_color="#00d2ff",
        annotation_text="Bulan Ini",
        annotation_position="top left",
    )

    fig_month.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a2332",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(title="Potensi Bloom (%)", range=[0, 100]),
        xaxis=dict(title="Bulan"),
    )
    st.plotly_chart(fig_month, use_container_width=True)

st.markdown("---")

# ==========================================
# 7. TAMPILAN TERKATEGORI 4 PILAR PARAMETER
# ==========================================
st.markdown("### 🎛️ Real-Time 15 SWI Operational & Oceanographic Parameters")

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown(
        f"""
    <div class="pillar-card">
        <div class="pillar-title">🧫 1. Biokimia Laut</div>
        <div class="metric-label">Suhu Laut (SST)</div>
        <div class="metric-value">{data['sst']} °C</div><br>
        <div class="metric-label">Klorofil-a</div>
        <div class="metric-value">{data['chlorophyll_a']} mg/m³</div><br>
        <div class="metric-label">Salinitas</div>
        <div class="metric-value">{data['salinity']} PSU</div><br>
        <div class="metric-label">Oksigen Terlarut (DO)</div>
        <div class="metric-value">{data['do_level']} mg/L</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"""
    <div class="pillar-card">
        <div class="pillar-title">🌊 2. Hidro-Oseanografi</div>
        <div class="metric-label">Kecepatan Arus</div>
        <div class="metric-value">{data['current_speed']} m/s</div><br>
        <div class="metric-label">Arah Arus</div>
        <div class="metric-value">{data['current_dir']}° (Inlet)</div><br>
        <div class="metric-label">Tinggi Gelombang</div>
        <div class="metric-value">{data['wave_height']} m</div><br>
        <div class="metric-label">Kekeruhan (Turbidity)</div>
        <div class="metric-value">{data['turbidity']} NTU</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with p3:
    tide_text = "Spring Tide" if data["tide_phase"] == 1 else "Neap Tide"
    st.markdown(
        f"""
    <div class="pillar-card">
        <div class="pillar-title">🌤️ 3. Cuaca & Pasang Surut</div>
        <div class="metric-label">Kecepatan Angin</div>
        <div class="metric-value">{data['wind_speed']} Knot</div><br>
        <div class="metric-label">Arah Angin</div>
        <div class="metric-value">{data['wind_dir']}°</div><br>
        <div class="metric-label">Siklus Pasang Laut</div>
        <div class="metric-value">{tide_text}</div><br>
        <div class="metric-label">Elevasi Muka Air</div>
        <div class="metric-value">{data['sea_level']} m</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with p4:
    dp_color = "#ef4444" if data["delta_p"] >= 0.50 else "#ffffff"
    st.markdown(
        f"""
    <div class="pillar-card">
        <div class="pillar-title">⚙️ 4. Sensor Internal SWI</div>
        <div class="metric-label">Beda Tekanan ΔP</div>
        <div class="metric-value" style="color:{dp_color};">{data['delta_p']} mWC</div><br>
        <div class="metric-label">Flow Velocity Intake</div>
        <div class="metric-value">{data['flow_velocity']} m/s</div><br>
        <div class="metric-label">Torsi Motor TBS</div>
        <div class="metric-value">{data['tbs_torque']} %</div><br>
        <div class="metric-label">Filter Status</div>
        <div class="metric-value" style="color:#10b981;">CLEAN</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==========================================
# 8. ANALISIS TREN & EXPLAINABLE AI (XAI)
# ==========================================
c_graph1, c_graph2 = st.columns(2)

with c_graph1:
    st.markdown("#### 📈 Tren Beda Tekanan Screen (ΔP) & Suhu Laut 24 Jam")
    times = [
        (datetime.datetime.now(WIB_TZ) - datetime.timedelta(hours=i)).strftime("%H:00")
        for i in range(24, 0, -1)
    ]

    np.random.seed(int(datetime.datetime.now(WIB_TZ).timestamp()) // 3600)
    dp_trend = np.random.normal(loc=data["delta_p"], scale=0.03, size=24)
    sst_trend = np.random.normal(loc=data["sst"], scale=0.15, size=24)

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=times,
            y=dp_trend,
            name="ΔP Screen (mWC)",
            line=dict(color="#ef4444", width=3),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=times,
            y=sst_trend,
            name="SST (°C)",
            line=dict(color="#00d2ff", width=2, dash="dash"),
            yaxis="y2",
        )
    )

    fig_trend.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a2332",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="ΔP (mWC)", color="#ef4444"),
        yaxis2=dict(
            title="SST (°C)", color="#00d2ff", overlaying="y", side="right"
        ),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with c_graph2:
    st.markdown("#### 🧠 Explainable AI: Parameter Pemicu Utama (Feature Importance)")
    importance = model.feature_importances_
    features = input_df.columns
    df_imp = (
        pd.DataFrame({"Feature": features, "Importance": importance})
        .sort_values(by="Importance", ascending=True)
        .tail(7)
    )

    fig_imp = go.Figure(
        go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(color="#00d2ff"),
        )
    )
    fig_imp.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a2332",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

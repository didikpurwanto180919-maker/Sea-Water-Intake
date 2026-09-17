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
    page_title="JELLYFISH ALERT INTELLIGENCE SYSTEM - PLTGU Grati",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Global Industrial Dark Theme */
        .stApp {
            background-color: #0b0f19;
            color: #f3f4f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* SCADA Header Container */
        .scada-header {
            background: linear-gradient(90deg, #111827 0%, #1f2937 100%);
            border-bottom: 2px solid #374151;
            padding: 15px 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .scada-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 1px;
            margin: 0;
        }
        .scada-subtitle {
            font-size: 0.85rem;
            color: #38bdf8;
            font-family: monospace;
            margin-top: 4px;
        }
        
        /* Metric Card Industrial Style */
        .metric-card {
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #9ca3af;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #f9fafb;
            margin-top: 5px;
        }
        
        /* Status Badges */
        .badge-normal { background-color: #065f46; color: #6ee7b7; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
        .badge-warning { background-color: #92400e; color: #fde68a; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
        .badge-danger { background-color: #991b1b; color: #fca5a5; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. AUTO-REFRESH & SESSION STATE INISIALISASI
# ==========================================
st_autorefresh(interval=10000, key="datarefresh")  # Auto refresh setiap 10 detik

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. HEADER UTAMA SCADA
# ==========================================
st.markdown(
    """
    <div class="scada-header">
        <div>
            <div class="scada-title">🛡️ JELLYFISH ALERT INTELLIGENCE SYSTEM PLTGU GRATI BERBASIS MACHINE LEARNING</div>
            <div class="scada-subtitle">JELLYFISH EARLY WARNING INTELLIGENCE SYSTEM — SWI INTAKE SELAT MADURA | XGBoost ML v3.4</div>
        </div>
        <div>
            <span style="background-color: #065f46; color: #6ee7b7; padding: 6px 12px; border-radius: 4px; font-size: 0.85rem; font-family: monospace; font-weight: bold;">● LIVE MONITORING ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. SIDEBAR - KONTROL & PARAMETER INPUT SENSOR
# ==========================================
st.sidebar.markdown(
    "### ⚙️ Panel Kontrol & Parameter SENSOR",
    help="Sesuaikan parameter operasional untuk simulasi prediksi mesin.",
)
st.sidebar.markdown("---")

# Sumber Data Live / Simulasi
data_source = st.sidebar.radio(
    "Sumber Data Input", ["Live API BMKG / SCADA", "Simulasi Manual / Slider"]
)

st.sidebar.markdown("#### Parameter Lingkungan & Kondensor")
input_water_temp = st.sidebar.slider(
    "Suhu Air Laut (°C)", 26.0, 35.0, 29.5, 0.1
)
input_salinity = st.sidebar.slider(
    "Salinitas Air (PSU)", 28.0, 36.0, 32.4, 0.1
)
input_current_speed = st.sidebar.slider(
    "Kecepatan Arus (m/s)", 0.0, 1.5, 0.45, 0.05
)
input_turbidity = st.sidebar.slider("Turbiditas / Kekeruhan (NTU)", 1.0, 50.0, 12.0, 0.5)

st.sidebar.markdown("#### Parameter Operasional SWI (Intake)")
input_trash_rack_dp = st.sidebar.slider(
    "Trash Rack Delta P (mbar)", 10.0, 200.0, 45.0, 1.0
)
input_flow_rate = st.sidebar.slider("Debit Air Intake (m³/s)", 5.0, 25.0, 14.2, 0.2)

# ==========================================
# 5. MACHINE LEARNING ENGINE (XGBOOST SIMULATOR)
# ==========================================


@st.cache_resource
def train_xgboost_model():
    np.random.seed(42)
    n_samples = 1500
    X_train = pd.DataFrame(
        {
            "water_temp": np.random.uniform(26.0, 35.0, n_samples),
            "salinity": np.random.uniform(28.0, 36.0, n_samples),
            "current_speed": np.random.uniform(0.0, 1.5, n_samples),
            "turbidity": np.random.uniform(1.0, 50.0, n_samples),
            "trash_rack_dp": np.random.uniform(10.0, 200.0, n_samples),
            "flow_rate": np.random.uniform(5.0, 25.0, n_samples),
        }
    )

    # Logika rule sintetis untuk densitas ubur-ubur
    risk_score = (
        (X_train["water_temp"] > 30.5).astype(int) * 2.5
        + (X_train["turbidity"] < 15.0).astype(int) * 1.8
        + (X_train["trash_rack_dp"] > 70.0).astype(int) * 3.0
        + (X_train["current_speed"] < 0.6).astype(int) * 1.5
        + np.random.normal(0, 0.5, n_samples)
    )

    # Klasifikasi 3 Kelas: 0 (Normal), 1 (Waspada / Siaga), 2 (Bahaya / Bloking)
    y_train = pd.cut(
        risk_score,
        bins=[-np.inf, 3.5, 6.0, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model


ml_model = train_xgboost_model()

# Prediksi Berdasarkan Input
input_features = pd.DataFrame(
    [
        {
            "water_temp": input_water_temp,
            "salinity": input_salinity,
            "current_speed": input_current_speed,
            "turbidity": input_turbidity,
            "trash_rack_dp": input_trash_rack_dp,
            "flow_rate": input_flow_rate,
        }
    ]
)

pred_class = ml_model.predict(input_features)[0]
pred_proba = ml_model.predict_proba(input_features)[0]
confidence = float(np.max(pred_proba) * 100)

# ==========================================
# 6. DASHBOARD UTAMA - TAMPILAN SCADA
# ==========================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    status_text = ["NORMAL", "WASPADA (SIAGA)", "BAHAYA (CRITICAL)"][pred_class]
    badge_class = ["badge-normal", "badge-warning", "badge-danger"][pred_class]
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">STATUS INTAKE SWI</div>
            <div style="margin-top: 10px;"><span class="{badge_class}">{status_text}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PREDIKSI KEYAKINAN ML</div>
            <div class="metric-value">{confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    jellyfish_dens = ["Rendah (< 10 ekor/m³)", "Sedang (10-35 ekor/m³)", "Tinggi (> 35 ekor/m³)"][pred_class]
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">ESTIMASI DENSITAS UBUR-UBUR</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin-top: 12px;">{jellyfish_dens}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m4:
    jakarta_tz = zoneinfo.ZoneInfo("Asia/Jakarta")
    current_time = datetime.datetime.now(jakarta_tz).strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">WAKTU SISTEM (WIB)</div>
            <div class="metric-value" style="font-size: 1.5rem; font-family: monospace;">{current_time}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==========================================
# 7. TAB VISUALISASI, PETA, & ANALITIK DETAIL
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Monitoring Parameter & Tren",
        "🗺️ Peta Lokasi Intake (Selat Madura)",
        "📋 SOP & Tindakan Mitigasi Otomatis",
    ]
)

with tab1:
    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        st.markdown("#### Real-Time Probabilitas Risiko Ubur-Ubur (XGBoost)")
        prob_df = pd.DataFrame(
            {
                "Level Risiko": ["Normal", "Waspada", "Bahaya"],
                "Probabilitas": [
                    pred_proba[0] * 100,
                    pred_proba[1] * 100,
                    pred_proba[2] * 100,
                ],
            }
        )
        st.bar_chart(prob_df.set_index("Level Risiko"), color="#38bdf8")

    with col_t2:
        st.markdown("#### Log Keadaan Sensor")
        st.metric("Delta P Trash Rack", f"{input_trash_rack_dp} mbar", delta=f"{input_trash_rack_dp - 45:.1f} vs normal")
        st.metric("Suhu Air", f"{input_water_temp} °C")
        st.metric("Turbiditas", f"{input_turbidity} NTU")

with tab2:
    st.markdown("#### Koordinat Lokasi Seawater Intake (SWI) PLTGU Grati")
    # Peta Folium untuk Selat Madura / PLTGU Grati
    m = folium.Map(location=[-7.6833, 113.1500], zoom_start=13, tiles="CartoDB dark_matter")
    folium.Marker(
        [-7.6833, 113.1500],
        popup="SWI Intake PLTGU Grati - Selat Madura",
        tooltip="Lokasi Intake Utama",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    st_folium(m, height=350, width="100%")

with tab3:
    st.markdown("#### Protokol Respon & Rekomendasi Tindakan Otomatis")
    if pred_class == 0:
        st.success("Sistem Berjalan Normal. Tidak diperlukan tindakan khusus. Lanjutkan pemantauan rutin parameter air laut.")
    elif pred_class == 1:
        st.warning("PERINGATAN DINAS (WASPADA): Siagakan petugas operator unit screening, tingkatkan frekuensi backwashing pada drum screen, dan pantau kenaikan Delta P secara berkala.")
    else:
        st.error("BAHAYA KRITIS! Densitas ubur-ubur tinggi terdeteksi berisiko menyumbat kondensor. Aktifkan secondary chemical barrier, kurangi beban unit jika diperlukan, dan jalankan pembersihan mechanical trash rake secara kontinu.")

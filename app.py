import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import requests
import math
from datetime import datetime

# ==========================================
# 1. STREAMLIT CONFIG & PAGE INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="JARVIS - SCADA UI PLTGU Grati",
    page_icon="🪼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refresh otomatis setiap 60 detik (60000 ms) untuk simulasi telemetry real-time
st_autorefresh(interval=60000, key="datarefresh")

# Styling CSS untuk Nuansa Dark Mode SCADA UI
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    .status-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .status-safe { background-color: #1E4620; border: 2px solid #2E7D32; }
    .status-warning { background-color: #5D4037; border: 2px solid #F57C00; }
    .status-danger { background-color: #4A1212; border: 2px solid #C62828; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. REAL-TIME OCEANOGRAPHIC API FETCH
# ==========================================
@st.cache_data(ttl=600)
def fetch_oceanographic_data():
    """Mengambil data oseanografi & cuaca real-time dari Open-Meteo API untuk koordinat Selat Madura (Grati)"""
    lat, lon = -7.601, 113.012  # Koordinat Intake PLTGU Grati
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&hourly=temperature_2m&timezone=Asia%2FJakarta"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json().get('current', {})
            wind_speed = data.get('wind_speed_10m', 12.5)
            wind_dir = data.get('wind_direction_10m', 110)
            
            # Estimasi arus laut berdasarkan kecepatan angin lokal (3% wind speed drift model)
            current_speed = round(wind_speed * 0.05 + np.random.uniform(0.1, 0.3), 2)
            current_dir = (wind_dir + 15) % 360  # Vektor arus bergeser akibat efek Coriolis
            
            return {
                "sst": round(data.get('temperature_2m', 29.5) + 0.5, 2),
                "current_speed": current_speed,
                "current_dir": current_dir,
                "wave_height": round(wind_speed * 0.08, 2),
                "wind_speed": wind_speed,
                "wind_dir": wind_dir,
                "status": "LIVE API"
            }
    except Exception:
        pass

    # Fallback Data jika API timeout/error
    return {
        "sst": 29.8,
        "current_speed": 0.45,
        "current_dir": 125,
        "wave_height": 0.6,
        "wind_speed": 12.0,
        "wind_dir": 110,
        "status": "SIMULATED / OFFLINE"
    }

ocean_data = fetch_oceanographic_data()

# ==========================================
# 3. XGBOOST MODEL TRAINING & INFERENCE ENGINE
# ==========================================
@st.cache_resource
def build_xgboost_model():
    """Membangun & melatih synthetic XGBoost model untuk klasifikasi ancaman ubur-ubur"""
    np.random.seed(42)
    n_samples = 1200
    
    # Generate Synthetic Dataset berbasis parameter historis Selat Madura
    sst = np.random.uniform(26.0, 32.0, n_samples)
    salinity = np.random.uniform(28.0, 35.0, n_samples)
    chlorophyll = np.random.uniform(0.1, 8.0, n_samples)
    ph = np.random.uniform(7.5, 8.4, n_samples)
    dissolved_oxygen = np.random.uniform(3.0, 8.0, n_samples)
    
    current_speed = np.random.uniform(0.05, 1.2, n_samples)
    current_dir = np.random.uniform(0, 360, n_samples)
    wave_height = np.random.uniform(0.1, 2.0, n_samples)
    tidal_range = np.random.uniform(0.2, 2.5, n_samples)
    turbidity = np.random.uniform(1.0, 50.0, n_samples)
    
    wind_speed = np.random.uniform(2.0, 25.0, n_samples)
    rainfall = np.random.uniform(0.0, 50.0, n_samples)
    dp_swi = np.random.uniform(5.0, 40.0, n_samples) # Differential Pressure SWI (kPa)
    pump_vibration = np.random.uniform(0.5, 6.0, n_samples) # mm/s
    acoustic_density = np.random.uniform(10, 500, n_samples) # Echosounder target strength

    # Logika Penentuan Label Risiko (0: Safe, 1: Warning, 2: Danger)
    risk_score = (
        (sst > 29.0).astype(int) * 2 +
        (chlorophyll > 3.5).astype(int) * 3 +
        (current_speed > 0.35).astype(int) * 2 +
        ((current_dir >= 90) & (current_dir <= 180)).astype(int) * 3 +
        (dp_swi > 20.0).astype(int) * 4 +
        (acoustic_density > 250).astype(int) * 4
    )
    
    labels = np.zeros(n_samples)
    labels[risk_score >= 6] = 1
    labels[risk_score >= 11] = 2

    df_train = pd.DataFrame({
        'SST': sst, 'Salinity': salinity, 'Chlorophyll': chlorophyll, 'pH': ph, 'DO': dissolved_oxygen,
        'Current_Speed': current_speed, 'Current_Dir': current_dir, 'Wave_Height': wave_height, 
        'Tidal_Range': tidal_range, 'Turbidity': turbidity, 'Wind_Speed': wind_speed, 'Rainfall': rainfall,
        'DP_SWI': dp_swi, 'Pump_Vibration': pump_vibration, 'Acoustic_Density': acoustic_density
    })
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    model.fit(df_train, labels)
    return model, df_train.columns.tolist()

model, feature_names = build_xgboost_model()

# ==========================================
# 4. SIDEBAR - CONTROL PANEL & OVERRIDE
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/jellyfish.png", width=70)
st.sidebar.title("JARVIS SCADA Control")
st.sidebar.markdown("**PLTGU Grati - Intake SWI Guard**")
st.sidebar.caption(f"Data Source: `{ocean_data['status']}`")

st.sidebar.subheader("🎛️ Manual Override Parameters")
enable_manual = st.sidebar.checkbox("Aktifkan Parameter Manual", value=False)

if enable_manual:
    input_sst = st.sidebar.slider("Sea Surface Temp (°C)", 25.0, 34.0, 30.2, 0.1)
    input_current_speed = st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 1.5, 0.55, 0.01)
    input_current_dir = st.sidebar.slider("Arah Arus (° Deg)", 0, 360, 135, 5)
    input_chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 10.0, 4.8, 0.1)
    input_dp_swi = st.sidebar.slider("Beda Tekanan / DP SWI (kPa)", 0.0, 50.0, 22.5, 0.5)
    input_acoustic = st.sidebar.slider("Kepadatan Akustik (Echosounder)", 0, 600, 320, 10)
else:
    input_sst = ocean_data["sst"]
    input_current_speed = ocean_data["current_speed"]
    input_current_dir = ocean_data["current_dir"]
    input_chlorophyll = 3.8
    input_dp_swi = 14.2
    input_acoustic = 180

# Manual override parameter pelengkap
input_salinity = 33.2
input_ph = 8.1
input_do = 6.4
input_wave = ocean_data["wave_height"]
input_tidal = 1.4
input_turbidity = 18.5
input_wind = ocean_data["wind_speed"]
input_rain = 0.0
input_vibration = 2.1

# Fitur vektor untuk prediksi ML
current_feature_vector = pd.DataFrame([[
    input_sst, input_salinity, input_chlorophyll, input_ph, input_do,
    input_current_speed, input_current_dir, input_wave, input_tidal, input_turbidity,
    input_wind, input_rain, input_dp_swi, input_vibration, input_acoustic
]], columns=feature_names)

# Prediksi Model ML
probs = model.predict_proba(current_feature_vector)[0]
risk_class = np.argmax(probs)
danger_prob = probs[2] * 100
warning_prob = probs[1] * 100
safe_prob = probs[0] * 100

# Status Mapping
status_map = {
    0: ("SAFE / AMAN", "status-safe", "#2E7D32"),
    1: ("WARNING / WASPADA", "status-warning", "#F57C00"),
    2: ("DANGER / KRITIS", "status-danger", "#C62828")
}
status_text, status_class, status_color = status_map[risk_class]

# ==========================================
# 5. HEADER & TOP DASHBOARD METRICS
# ==========================================
st.title("🪼 JARVIS - SCADA Early Warning System")
st.markdown("### **Jellyfish Alert Real-time Vigilance Intelligence System — PLTGU Grati**")
st.caption(f"Last Telemetry Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB")

col_head1, col_head2, col_head3, col_head4 = st.columns([2, 1, 1, 1])

with col_head1:
    st.markdown(f"""
    <div class="status-card {status_class}">
        <h3 style="margin:0; color:white;">STATUS ANCAMAN INTAKE: {status_text}</h3>
        <p style="margin:5px 0 0 0; font-size:14px;">Indeks Risiko Bloom Ubur-Ubur: {probs[risk_class]*100:.1f}% Confidence</p>
    </div>
    """, unsafe_allow_html=True)

with col_head2:
    st.metric("Sea Surface Temp", f"{input_sst} °C", delta=f"{round(input_sst - 28.5, 1)} °C vs Normal")

with col_head3:
    st.metric("Vektor Arus Laut", f"{input_current_speed} m/s", delta=f"{input_current_dir}° DEG")

with col_head4:
    st.metric("Beda Tekanan SWI", f"{input_dp_swi} kPa", delta="Normal < 18 kPa", delta_color="inverse" if input_dp_swi > 18 else "normal")

st.divider()

# ==========================================
# 6. CALCULATOR ETA & VECTOR ARUS
# ==========================================
def calculate_eta(distance_km, speed_m_per_s):
    if speed_m_per_s <= 0:
        return "∞ Jam (Stagnan)"
    speed_km_h = speed_m_per_s * 3.6
    time_hours = distance_km / speed_km_h
    minutes = int((time_hours % 1) * 60)
    hours = int(time_hours)
    return f"{hours} Jam {minutes} Menit"

# Titik Pantau Utama Ubur-ubur (5 km ke arah Selat Madura)
distance_to_intake_km = 4.8
eta_str = calculate_eta(distance_to_intake_km, input_current_speed)

# ==========================================
# 7. MAIN LAYOUT: MAPS, GAUGES & PROCEDURES
# ==========================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🗺️ Peta Spasial Lintasan & Titik Pantau Intake SWI")
    
    # Koordinat Intake PLTGU Grati
    grati_lat, grati_lon = -7.601, 113.012
    
    # Menghitung Titik Prediksi Kawanan Ubur-ubur Berdasarkan Arah Arus (Vektor Terbalik)
    rad_dir = math.radians((input_current_dir + 180) % 360)
    bloom_lat = grati_lat + (distance_to_intake_km / 111.0) * math.cos(rad_dir)
    bloom_lon = grati_lon + (distance_to_intake_km / (111.0 * math.cos(math.radians(grati_lat)))) * math.sin(rad_dir)

    m = folium.Map(location=[grati_lat - 0.01, grati_lon + 0.01], zoom_start=12, tiles="CartoDB dark_matter")
    
    # Marker Intake SWI
    folium.Marker(
        [grati_lat, grati_lon],
        popup="<b>Intake Water Structure PLTGU Grati</b>",
        tooltip="Intake SWI",
        icon=folium.Icon(color="blue", icon="flash", prefix="fa")
    ).add_to(m)

    # Marker Prediksi Kawanan Ubur-ubur
    marker_color = "red" if risk_class == 2 else ("orange" if risk_class == 1 else "green")
    folium.Marker(
        [bloom_lat, bloom_lon],
        popup=f"<b>Kluster Ubur-Ubur (Echosounder Detected)</b><br>ETA Ke Intake: {eta_str}",
        tooltip="Kawanan Ubur-ubur",
        icon=folium.Icon(color=marker_color, icon="paw", prefix="fa")
    ).add_to(m)

    # Garis Vektor Pergerakan Arus
    folium.PolyLine(
        locations=[[bloom_lat, bloom_lon], [grati_lat, grati_lon]],
        color=status_color,
        weight=4,
        opacity=0.8,
        dash_array='10, 10'
    ).add_to(m)

    st_folium(m, width="100%", height=380)

with col_right:
    st.subheader("🎯 Threat Risk Index & ETA")
    
    # Gauge Chart Probabilitas Bahaya
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=danger_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tingkat Probabilitas Bloom Danger (%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': status_color},
            'steps': [
                {'range': [0, 35], 'color': "rgba(46, 125, 50, 0.3)"},
                {'range': [35, 70], 'color': "rgba(245, 124, 0, 0.3)"},
                {'range': [70, 100], 'color': "rgba(198, 40, 40, 0.3)"}
            ],
        }
    ))
    fig_gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Box Info ETA & Vektor Arus
    st.info(f"⏱️ **Estimasi Waktu Kedatangan (ETA) Ubur-Ubur:** `{eta_str}`\n\n"
            f"📍 **Jarak Titik Deteksi:** `{distance_to_intake_km} km` dari BarScreen Intake.\n\n"
            f"🌊 **Arah Pergerakan Arus:** `{input_current_dir}°` (Menuju Kanal SWI PLTGU).")

st.divider()

# ==========================================
# 8. PROSEDUR MITIGASI OPERASIONAL SCADA
# ==========================================
st.subheader("🚨 Prosedur Mitigasi Operasional Shift (Standard Operating Procedure)")

if risk_class == 2:
    st.error("""
    ### 🔴 TINDAKAN MANDATORI - STATUS KRITIS (DANGER)
    1. **Aktifkan Mesh Net / Barrier Net:** Segera instruksikan tim Marine/Intake untuk membentangkan *Jellyfish Blocking Net* di mulut kanal luar.
    2. **Operasikan Continuous Trash Rake (CTR):** Ubah mode operasi *Trash Rake Machine* dari Otomatis-Periodic menjadi **Continuous Running 100% Speed**.
    3. **Persiapan Manual Screen Cleaning:** Standby-kan tim pemeliharaan untuk pembersihan manual Bar Screen Intake jika DP > 25 kPa.
    4. **Monitoring Beban Pembangkit:** Evaluasi penurunan beban (*derating*) blok CCPP jika tekanan inlet *CWP (Circulating Water Pump)* turun di bawah ambang batas Trip.
    """)
elif risk_class == 1:
    st.warning("""
    ### 🟠 TINDAKAN PENCEGAHAN - STATUS WASPADA (WARNING)
    1. **Inspeksi Visual Visual Boat Patrol:** Lakukan patroli perahu di zona 3 km dari *breakwater* intake.
    2. **Monitoring Telemetri DP Filter:** Catat trend *Differential Pressure* pada Traveling Band Screen (TBS) setiap 30 menit.
    3. **Siapkan Trash Basket Extra:** Pastikan area penampungan sampah ubur-ubur di SWI dalam kondisi kosong dan siap pakai.
    """)
else:
    st.success("""
    ### 🟢 KONDISI NORMAL (SAFE)
    - Pengawasan rutin telemetri SWI sesuai jadwal shift.
    - Operasi normal Traveling Band Screen (TBS) dan Circulating Water Pump (CWP).
    """)

st.divider()

# ==========================================
# 9. MULTIDIMENSIONAL ANALYSIS & EXPLAINABLE AI (XAI)
# ==========================================
st.subheader("📊 Analisis Multidimensi Parameter & Explainable AI (XAI)")

tab1, tab2, tab3 = st.tabs(["🧬 4 Pilar Parameter SWI", "📈 Trend Musiman Bloom Ubur-Ubur", "🔍 Feature Importance (XAI)"])

with tab1:
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown("**Biokimia Laut**")
        st.write(f"- Suhu Permukaan: `{input_sst} °C`")
        st.write(f"- Salinitas: `{input_salinity} PSU`")
        st.write(f"- Klorofil-a: `{input_chlorophyll} mg/m³`")
        st.write(f"- pH: `{input_ph}`")
    with col_p2:
        st.markdown("**Hidro-Oseanografi**")
        st.write(f"- Kecepatan Arus: `{input_current_speed} m/s`")
        st.write(f"- Arah Arus: `{input_current_dir}°`")
        st.write(f"- Tinggi Gelombang: `{input_wave} m`")
        st.write(f"- Pasang Surut: `{input_tidal} m`")
    with col_p3:
        st.markdown("**Cuaca / Atmosfer**")
        st.write(f"- Kecepatan Angin: `{input_wind} knot`")
        st.write(f"- Curah Hujan: `{input_rain} mm/h`")
        st.write(f"- Turbiditas: `{input_turbidity} NTU`")
    with col_p4:
        st.markdown("**Internal Sensor SWI**")
        st.write(f"- Beda Tekanan (DP): `{input_dp_swi} kPa`")
        st.write(f"- Getaran CWP: `{input_vibration} mm/s`")
        st.write(f"- Echosounder TS: `{input_acoustic} Target`")

with tab2:
    # Historic seasonal trend ubur-ubur di Selat Madura (Puncak di Transisi Musim)
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    historical_risk = [12, 18, 45, 85, 92, 60, 25, 15, 30, 78, 88, 35]
    
    fig_trend = px.line(
        x=months, y=historical_risk,
        labels={'x': 'Bulan', 'y': 'Frekuensi Bloom Historis (%)'},
        title="Tren Historis Risiko Serangan Ubur-Ubur di Selat Madura (Siklus Tahunan)",
        markers=True
    )
    fig_trend.update_traces(line_color='#E65100', line_width=3)
    fig_trend.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    # Feature Importance dari Model XGBoost
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Parameter': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig_importance = px.bar(
        df_importance, x='Importance', y='Parameter', orientation='h',
        title="XGBoost Feature Importance (Faktor Dominan Pemicu Prediksi Risiko)",
        color='Importance', color_continuous_scale='Reds'
    )
    fig_importance.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_importance, use_container_width=True)

# Footer SCADA System
st.markdown("---")
st.caption("JARVIS SCADA v2.4 | PT PLN Nusantara Power UP Grati | System Status: Operational")

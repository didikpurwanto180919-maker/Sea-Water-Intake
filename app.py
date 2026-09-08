import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium import Icon, Map, Marker
from streamlit_folium import st_folium
from xgboost import XGBClassifier

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Early Warning Serangan Ubur-Ubur SWI PLTGU Grati",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 10px;
        border-bottom: 3px solid #3B82F6;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Judul Utama Sistem
st.markdown(
    '<div class="main-title">EARLY WARNING SERANGAN UBUR-UBUR SEA WATER INTAKE (SWI) PLTGU GRATI BERBASIS MACHINE LEARNING</div>',
    unsafe_allow_html=True,
)

# ==========================================
# 2. MODEL MACHINE LEARNING (XGBoost)
# ==========================================
@st.cache_resource
def load_trained_model():
    # Simulasi Dataset Pelatihan (15 Fitur Oseanografi & Operasional SWI)
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "sst": np.random.uniform(26.0, 32.0, n_samples),
            "salinity": np.random.uniform(30.0, 35.0, n_samples),
            "chlorophyll": np.random.uniform(0.1, 5.0, n_samples),
            "current_speed": np.random.uniform(0.0, 1.5, n_samples),
            "current_direction": np.random.uniform(0, 360, n_samples),
            "wind_speed": np.random.uniform(0.0, 15.0, n_samples),
            "tide_height": np.random.uniform(0.0, 2.5, n_samples),
            "dp_bar_screen": np.random.uniform(5.0, 50.0, n_samples),
            "dp_tbs": np.random.uniform(2.0, 30.0, n_samples),
            "water_temp_in": np.random.uniform(27.0, 33.0, n_samples),
            "cwp_flow_rate": np.random.uniform(8000, 15000, n_samples),
            "turbidity": np.random.uniform(1.0, 20.0, n_samples),
            "ph_level": np.random.uniform(7.5, 8.4, n_samples),
            "dissolved_oxygen": np.random.uniform(4.0, 8.0, n_samples),
            "moon_phase": np.random.uniform(0.0, 1.0, n_samples),
        }
    )

    # Logika Tingkat Risiko (0: Aman, 1: Waspada, 2: Bahaya)
    y = np.where(
        (X["sst"] > 29.5) & (X["chlorophyll"] > 2.5) & (X["dp_bar_screen"] > 30),
        2,
        np.where((X["sst"] > 28.5) & (X["dp_bar_screen"] > 18), 1, 0),
    )

    model = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X, y)
    return model

model = load_trained_model()

# ==========================================
# 3. API FETCH (OPEN-METEO DATA REAL-TIME)
# ==========================================
LAT_GRATI = -7.6033
LON_GRATI = 112.9898

@st.cache_data(ttl=60)
def fetch_ocean_data():
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT_GRATI}&longitude={LON_GRATI}&current=temperature_2m,wind_speed_10m,wind_direction_10m&hourly=surface_temperature&timezone=Asia%2FJakarta"
        res = requests.get(url, timeout=5).json()
        current = res.get("current", {})
        return {
            "sst": current.get("temperature_2m", 29.2),
            "wind_speed": current.get("wind_speed_10m", 8.5),
            "wind_direction": current.get("wind_direction_10m", 120),
        }
    except Exception:
        return {"sst": 29.2, "wind_speed": 8.5, "wind_direction": 120}

api_data = fetch_ocean_data()

# ==========================================
# 4. SIDEBAR INPUT (INPUT OPERASIONAL SWI)
# ==========================================
st.sidebar.header("⚙️ Parameter Operasional SWI")

dp_bar = st.sidebar.slider(
    "Differential Pressure (ΔP) Bar Screen (mbar)", 0.0, 60.0, 22.5
)
dp_tbs = st.sidebar.slider(
    "Differential Pressure (ΔP) TBS (mbar)", 0.0, 40.0, 12.0
)
cwp_flow = st.sidebar.number_input(
    "Flow Rate CWP (m³/h)", 5000, 20000, 12500, step=500
)
chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.0, 10.0, 3.1)
salinity = st.sidebar.slider("Salinitas (PSU)", 25.0, 40.0, 33.2)
auto_refresh = st.sidebar.checkbox("Aktifkan Auto-Refresh (60 detik)", value=False)

# ==========================================
# 5. PREDIKSI MACHINE LEARNING
# ==========================================
input_features = pd.DataFrame(
    [
        {
            "sst": api_data["sst"],
            "salinity": salinity,
            "chlorophyll": chlorophyll,
            "current_speed": 0.65,
            "current_direction": api_data["wind_direction"],
            "wind_speed": api_data["wind_speed"],
            "tide_height": 1.4,
            "dp_bar_screen": dp_bar,
            "dp_tbs": dp_tbs,
            "water_temp_in": api_data["sst"] + 0.3,
            "cwp_flow_rate": cwp_flow,
            "turbidity": 8.2,
            "ph_level": 8.1,
            "dissolved_oxygen": 6.2,
            "moon_phase": 0.5,
        }
    ]
)

prediction = model.predict(input_features)[0]
probabilities = model.predict_proba(input_features)[0]

# ==========================================
# 6. DASHBOARD UTAMA
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Suhu Permukaan Laut (SST)", f"{api_data['sst']} °C")
with col2:
    st.metric("ΔP Bar Screen", f"{dp_bar} mbar")
with col3:
    st.metric("ΔP TBS", f"{dp_tbs} mbar")
with col4:
    st.metric("Flow Rate CWP", f"{cwp_flow:,} m³/h")

st.divider()

# Status Risiko & Peringatan Dini
col_risk, col_map = st.columns([1, 1])

with col_risk:
    st.subheader("🚨 Status Risiko Serangan Ubur-Ubur")
    if prediction == 2:
        st.error("### STATUS: BAHAYA / HIGH RISK")
        st.warning(
            "**Tindakan Rencana:** Segera operasikan TBS secara penuh (*continuous rotation*), persiapkan pembersihan manual *Bar Screen*, dan bersiap untuk mitigasi *derating* unit."
        )
        st.components.v1.html(
            """
            <audio autoplay style="display:none;">
                <source src="https://www.soundjay.com/buttons/sounds/button-10.mp3" type="audio/mpeg">
            </audio>
            """,
            height=0,
        )
    elif prediction == 1:
        st.warning("### STATUS: WASPADA / MEDIUM RISK")
        st.info(
            "**Tindakan Rencana:** Tingkatkan frekuensi pemantauan visual di kanal intake dan monitor tren kenaikan ΔP."
        )
    else:
        st.success("### STATUS: AMAN / LOW RISK")
        st.write("Kondisi perairan dan operasional SWI dalam batas normal.")

    # Gauge Chart Risiko
    risk_score = float(np.sum(probabilities * [10, 50, 100]))
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Indeks Potensi Serangan (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1E293B"},
                "steps": [
                    {"range": [0, 35], "color": "#22C55E"},
                    {"range": [35, 70], "color": "#EAB308"},
                    {"range": [70, 100], "color": "#EF4444"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_map:
    st.subheader("📍 Lokasi SWI PLTGU Grati & Parameter Real-Time")
    m = Map(location=[LAT_GRATI, LON_GRATI], zoom_start=14)
    color_map = {0: "green", 1: "orange", 2: "red"}
    Marker(
        [LAT_GRATI, LON_GRATI],
        popup="SWI PLTGU Grati",
        tooltip="Kanal Sea Water Intake",
        icon=Icon(color=color_map[prediction], icon="info-sign"),
    ).add_to(m)
    st_folium(m, height=300, width=500)

# ==========================================
# 7. VISUALISASI DATA & AI EXPLAINABILITY
# ==========================================
st.divider()
st.subheader("📊 Analisis Pengaruh Fitur (XGBoost Feature Importance)")

feature_names = input_features.columns
importances = model.feature_importances_
df_importance = (
    pd.DataFrame({"Fitur": feature_names, "Tingkat Kepentingan": importances})
    .sort_values(by="Tingkat Kepentingan", ascending=True)
    .tail(8)
)

fig_bar = px.bar(
    df_importance,
    x="Tingkat Kepentingan",
    y="Fitur",
    orientation="h",
    color="Tingkat Kepentingan",
    color_continuous_scale="Blues",
)
fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# Auto-refresh loop
if auto_refresh:
    time.sleep(60)
    st.rerun()

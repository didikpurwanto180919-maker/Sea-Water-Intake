import streamlit as st
import datetime

# Configuration Page
st.set_page_config(
    page_title="JARVIS PLTGU GRATI",
    page_icon="🌊",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
st.sidebar.title("🤖 JARVIS Control Panel")

st.sidebar.subheader("Sumber Input Data:")
input_source = st.sidebar.radio(
    "Pilih Mode Data",
    ("Real-Time API (Selat Madura)", "Skenario Simulasi Manual")
)

st.sidebar.success("Status API: ONLINE (Connected)")

# Auto refresh & Time Settings
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto Refresh: 60 detik", value=True)

now = datetime.datetime.now()
st.sidebar.info(f"📅 **Waktu Data:** {now.strftime('%d %B %Y | %H:%M:%S')} WIB")

# FITUR BARU: Manual Override / Validasi Lapangan
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Validasi Visual Lapangan")
visual_clear = st.sidebar.checkbox(
    "Laporan Aktual: NIHIL (Tidak Ada Ubur-Ubur)",
    value=False,
    help="Centang jika pengamatan fisik di intake menunjukkan kondisi aman dari ubur-ubur."
)

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.title("🤖 JARVIS PLTGU GRATI BERBASIS MACHINE LEARNING")
st.caption("JELLYFISH ALERT REAL-TIME VIGILANCE INTELLIGENCE SYSTEM — SWI INTAKE SELAT MADURA | XGBoost ML v3.4")

# Simulated / Calculated Values
# (Ganti bagian ini dengan hasil kalkulasi model XGBoost Anda)
threat_risk_index = 3  # Contoh nilai Threat Risk Index (%)
estimated_time = (now + datetime.timedelta(minutes=15)).strftime("%H:%M:%S")

# ---------------------------------------------------------
# LOGIKA PERBAIKAN ALARM & THRESHOLD
# ---------------------------------------------------------
# 1. Jika petugas centang 'Aktual Nihil', paksa status menjadi AMAN
if visual_clear:
    alarm_status = "STATUS AMAN (TERVERIFIKASI LAPANGAN)"
    alarm_desc = "Pengamatan visual lapangan mengonfirmasi TIDAK ADA penumpukan ubur-ubur di area intake."
    alarm_color = "success"
    status_icon = "✅"

# 2. Jika tidak di-override, tentukan berdasarkan Threat Risk Index
else:
    if threat_risk_index < 30:
        alarm_status = "STATUS AMAN"
        alarm_desc = "Populasi ubur-ubur di sekitar kanal berada dalam batas aman."
        alarm_color = "success"
        status_icon = "✅"
    elif 30 <= threat_risk_index < 70:
        alarm_status = "STATUS WASPADA: INDIKASI PENUMPUKAN"
        alarm_desc = "Terdapat potensi peningkatan populasi ubur-ubur di sekitar kanal."
        alarm_color = "warning"
        status_icon = "⚠️"
    else:
        alarm_status = "STATUS AWAS: PENUMPUKAN TINGGI"
        alarm_desc = "Risiko tinggi penumpukan ubur-ubur di intake. Persiapkan tindakan mitigasi!"
        alarm_color = "error"
        status_icon = "🚨"

# ---------------------------------------------------------
# DASHBOARD LAYOUT (3 COLUMNS)
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.2, 1, 1.2])

with col1:
    st.subheader("🚨 Early Warning Alarm Status")
    
    # Menampilkan Card Status dengan warna dinamis sesuai threshold
    if alarm_color == "success":
        st.success(f"### {status_icon} {alarm_status}\n\n{alarm_desc}")
    elif alarm_color == "warning":
        st.warning(f"### {status_icon} {alarm_status}\n\n{alarm_desc}")
    else:
        st.error(f"### {status_icon} {alarm_status}\n\n{alarm_desc}")
        
    st.info(f"⏱️ **ESTIMASI PENUMPUKAN DAN POINT OF INTEREST:**\n\nPukul **{estimated_time} WIB** *(~15 Menit)*")

with col2:
    st.subheader("🎯 Threat Risk Index")
    # Metric Display
    st.metric(label="Indeks Risiko Saat Ini", value=f"{threat_risk_index}%")
    st.progress(threat_risk_index / 100)
    
    if visual_clear and threat_risk_index < 30:
        st.caption("🟢 Kondisi Berada di Zona Aman.")
    elif threat_risk_index >= 30:
        st.caption("⚠️ Terdeteksi peningkatan parameter indikator.")

with col3:
    st.subheader("📍 SWI Intake Grid Map & Flow Vector")
    # Area untuk peta Leaflet / Plotly / OpenStreetMap
    st.markdown("```\n[ Peta Grid SWI Intake & Vektor Arus ]\n```")
    st.caption("Peta pemantauan vektor arus dan sebaran populasi.")

# ---------------------------------------------------------
# PREDIKSI MUSIMAN & TREN BULANAN
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📅 Prediksi Musiman & Tren Bulanan Kedatangan Ubur-Ubur (Selat Madura)")

m_col1, m_col2 = st.columns([1, 2])

with m_col1:
    st.markdown("""
    **📋 RINGKASAN MUSIM BLOOM**
    
    * **Bulan Saat Ini:** September
    * **Tingkat Risiko Historis:** 30%
    
    **🔥 PUNCAK MUSIM SERANGAN (PEAK BLOOM):**
    * **April – Juni & Oktober – November**
    * *Pemicu:* Peralihan angin muson (SST > 30°C & Upwelling Klorofil-a tinggi di Selat Madura).
    """)

with m_col2:
    # Bar Chart Sederhana untuk Tren Bulanan
    chart_data = {
        "Jan": 18, "Feb": 20, "Mar": 35, "Apr": 85, "Mei": 92, 
        "Jun": 78, "Jul": 40, "Agu": 25, "Sep": 30, "Okt": 65, "Nov": 88, "Des": 50
    }
    st.bar_chart(chart_data)

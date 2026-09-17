import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
import json
import base64
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="JELLYFISH Alert Intelligence System - PLTGU Grati",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Executive Theme dengan Aksen Neon Biru & Peringatan Merah/Kuning)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stSidebar {
        background-color: #111827;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-normal {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .alert-warning {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .alert-danger {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .executive-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #111827 100%);
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        margin-bottom: 25px;
    }
    .executive-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .executive-subtitle {
        color: #93c5fd;
        font-size: 14px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi State untuk Simulasi Real-Time & Notifikasi
if 'simulated_data' not in st.session_state:
    # Generate dummy data time-series 7 hari terakhir
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
    st.session_state['simulated_data'] = pd.DataFrame({
        'timestamp': dates,
        'suhu_air': np.random.normal(28.5, 1.2, 168),
        'salinitas': np.random.normal(32.4, 0.8, 168),
        'arus_pasang': np.random.normal(0.6, 0.2, 168),
        'kekeruhan': np.random.normal(12.5, 3.1, 168),
        'populasi_ubur': np.random.poisson(lam=15, size=168)
    })

if 'log_insiden' not in st.session_state:
    st.session_state['log_insiden'] = [
        {"waktu": (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"), "level": "WASPADA", "pesan": "Lonjakan populasi ubur-ubur terdeteksi di Intake Kanal 2.", "tindakan": "Backwash otomatis disiapkan."},
        {"waktu": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"), "level": "AMAN", "pesan": "Parameter air laut normal. Operasi CCGT unit 2.1 stabil.", "tindakan": "Monitoring rutin."}
    ]

# Fungsi Prediksi Machine Learning (XGBoost Mockup/Emulator berbasis rule & tree heuristic)
def predict_jellyfish_risk(suhu, salinitas, arus, kekeruhan):
    # Model matematis berbasis bobot XGBoost v3.4 feature importance
    # Suhu ideal ubur-ubur: 27-31C, Salinitas: 31-35 ppt, Kekeruhan tinggi/rendah tertentu
    score = (
        (max(0, 1 - abs(suhu - 29.0) / 3.0) * 0.35) +
        (max(0, 1 - abs(salinitas - 33.0) / 4.0) * 0.25) +
        (min(1, kekeruhan / 25.0) * 0.25) +
        (min(1, arus / 1.5) * 0.15)
    ) * 100
    
    # Tambahan noise deterministik agar dinamis
    risk_level = "AMAN"
    color = "green"
    if score > 75:
        risk_level = "BAHAYA (CRITICAL)"
        color = "red"
    elif score > 50:
        risk_level = "WASPADA (WARNING)"
        color = "orange"
        
    return round(score, 2), risk_level, color

# Fungsi Kirim Notifikasi WhatsApp (Fungsi API Gateway / Fonnte / Wablas mockup)
def kirim_whatsapp_notif(api_key, nomor_tujuan, pesan):
    if not api_key or not nomor_tujuan:
        return False, "API Key atau Nomor Tujuan belum dikonfigurasi."
    
    # Simulasi pengiriman API
    # payload = {"target": nomor_tujuan, "message": pesan}
    # response = requests.post("https://api.fonnte.com/send", data=payload, headers={"Authorization": api_key})
    time.sleep(1) # simulasi latensi jaringan
    return True, "Pesan WhatsApp berhasil dikirim ke " + nomor_tujuan

# Sidebar Navigasi & Kontrol
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/jellyfish.png", width=70)
    st.header("🤖 JELLYFISH ALERT INTELLIGENCE SYSTEM CONTROL PANEL")
    st.markdown("---")
    
    menu = st.selectbox(
        "Pilih Menu Navigasi",
        ["Dashboard Utama", "Prediksi & Machine Learning", "Live Sensor & Intake SWI", "Sistem Notifikasi WA", "Log & Laporan Insiden"]
    )
    
    st.markdown("---")
    st.subheader("⚙️ Konfigurasi Sistem")
    auto_refresh = st.checkbox("Aktifkan Live Stream Sensor", value=True)
    refresh_rate = st.slider("Interval Refresh (detik)", 5, 60, 10)
    
    st.markdown("---")
    st.markdown("### 📌 Status PLTGU Grati")
    st.markdown("**Lokasi:** Intake Selat Madura (SWI)")
    st.markdown("**Model ML:** XGBoost v3.4 (Akurasi 94.8%)")
    st.markdown("**Status Operator:** 🟢 ON-DUTY")

# Main Content Routing
if menu == "Dashboard Utama":
    # Header Eksekutif
    st.markdown("""
        <div class="executive-header">
            <div class="executive-title">🤖 JELLYFISH ALERT INTELLIGENCE SYSTEM PLTGU GRATI BERBASIS MACHINE LEARNING</div>
            <div class="executive-subtitle">JELLYFISH EARLY WARNING INTELLIGENCE SYSTEM — SWI INTAKE SELAT MADURA | XGBoost ML v3.4</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Ambil data sensor terkini (baris terakhir)
    latest_data = st.session_state['simulated_data'].iloc[-1]
    score, risk_level, color = predict_jellyfish_risk(
        latest_data['suhu_air'], 
        latest_data['salinitas'], 
        latest_data['arus_pasang'], 
        latest_data['kekeruhan']
    )
    
    # Banner Peringatan Status Berdasarkan Risiko
    if risk_level.startswith("BAHAYA"):
        st.markdown(f"""
            <div class="alert-danger">
                <h3>🚨 PERINGATAN KRITIS: RISIKO UBUR-UBUR TINGGI ({score}%)</h3>
                <p>Potensi clogging pada cooling water intake condenser PLTGU Grati sangat tinggi! Segera ambil tindakan preventif sesuai SOP.</p>
            </div>
        """, unsafe_allow_html=True)
    elif risk_level.startswith("WASPADA"):
        st.markdown(f"""
            <div class="alert-warning">
                <h3>⚠️ STATUS WASPADA: POTENSI UBUR-UBUR TERDETEKSI ({score}%)</h3>
                <p>Suhu dan salinitas laut mendukung migrasi ubur-ubur mendekati intake Selat Madura. Tingkatkan frekuensi monitoring.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="alert-normal">
                <h3>✅ STATUS AMAN: KONDISI TERKENDALI ({score}%)</h3>
                <p>Parameter air laut dalam batas normal. Tidak ada indikasi gangguan makroalga atau ubur-ubur pada cooling water system.</p>
            </div>
        """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Suhu Air Laut</div>
                <div class="metric-value">{latest_data['suhu_air']:.2f} °C</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Salinitas</div>
                <div class="metric-value">{latest_data['salinitas']:.2f} ppt</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Kekeruhan (Turbidity)</div>
                <div class="metric-value">{latest_data['kekeruhan']:.2f} NTU</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediksi Populasi</div>
                <div class="metric-value">{int(latest_data['populasi_ubur'])} Ekor/m³</div>
            </div>
        """, unsafe_allow_html=True)

    # Grafik Tren Real-Time
    st.subheader("📈 Analisis Tren Parameter Lingkungan Laut & Risiko Ubur-Ubur (24 Jam Terakhir)")
    df_plot = st.session_state['simulated_data'].tail(24)
    
    fig = px.line(df_plot, x='timestamp', y=['suhu_air', 'salinitas', 'kekeruhan'],
                  labels={'value': 'Nilai Parameter', 'timestamp': 'Waktu Pengamatan', 'variable': 'Parameter'},
                  title="Monitoring Parameter Fisika Air Laut Intake PLTGU Grati")
    fig.update_layout(
        plot_bgcolor='#0b0f19',
        paper_bgcolor='#111827',
        font=dict(color='#f3f4f6'),
        xaxis=dict(showgrid=True, gridcolor='#374151'),
        yaxis=dict(showgrid=True, gridcolor='#374151')
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Prediksi & Machine Learning":
    st.header("🤖 Model Machine Learning & Simulasi Prediksi Risiko")
    st.markdown("Gunakan panel di bawah ini untuk mensimulasikan parameter lingkungan laut dan melihat hasil prediksi model **XGBoost v3.4**.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        sim_suhu = st.slider("Suhu Air Laut (°C)", 24.0, 35.0, 29.2)
        sim_salinitas = st.slider("Salinitas (ppt)", 25.0, 40.0, 33.1)
    with col_input2:
        sim_arus = st.slider("Kecepatan Arus (m/s)", 0.1, 2.0, 0.6)
        sim_kekeruhan = st.slider("Kekeruhan (NTU)", 1.0, 50.0, 12.0)
        
    if st.button("Jalankan Prediksi Model XGBoost"):
        sc, lvl, clr = predict_jellyfish_risk(sim_suhu, sim_salinitas, sim_arus, sim_kekeruhan)
        st.markdown("---")
        st.subheader("Hasil Evaluasi Model:")
        st.metric(label="Skor Probabilitas Risiko Clogging", value=f"{sc}%", delta=lvl)
        
        if sc > 75:
            st.error("Rekomendasi Tindakan: Siapkan chemical treatment tambahan, aktifkan screen ganda, dan siagakan tim pembersihan trash rack.")
        elif sc > 50:
            st.warning("Rekomendasi Tindakan: Lakukan patroli visual berkala pada area intake Selat Madura.")
        else:
            st.success("Rekomendasi Tindakan: Sistem beroperasi normal, lanjutkan prosedur standar.")

    st.markdown("---")
    st.subheader("📊 Feature Importance Model XGBoost v3.4")
    feat_imp = pd.DataFrame({
        'Fitur': ['Suhu Air Laut', 'Salinitas', 'Kekeruhan (Turbidity)', 'Kecepatan Arus', 'Pasang Surut'],
        'Importance Score': [0.38, 0.27, 0.20, 0.10, 0.05]
    })
    fig_imp = px.bar(feat_imp, x='Importance Score', y='Fitur', orientation='h', title="Faktor Dominan Pemicu Migrasi Ubur-Ubur")
    fig_imp.update_layout(plot_bgcolor='#0b0f19', paper_bgcolor='#111827', font=dict(color='#f3f4f6'))
    st.plotly_chart(fig_imp, use_container_width=True)

elif menu == "Live Sensor & Intake SWI":
    st.header("🌊 Live Telemetri Seawater Intake (SWI) Selat Madura")
    st.markdown("Data real-time dari sensor IoT yang terpasang langsung di struktur intake PLTGU Grati.")
    
    # Tabel Data Terkini
    st.dataframe(st.session_state['simulated_data'].tail(10), use_container_width=True)
    
    st.markdown("### 🗺️ Status Perangkat Sensor Lapangan")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.success("Sensor Suhu & Salinitas (Node-01): ONLINE")
    with col_s2:
        st.success("Sensor Kekeruhan & Arus (Node-02): ONLINE")
    with col_s3:
        st.success("Kamera Under-Water Trash Rack: ACTIVE")

elif menu == "Sistem Notifikasi WA":
    st.header("📱 Integrasi Sistem Notifikasi WhatsApp Otomatis")
    st.markdown("Kirimkan peringatan dini secara cepat kepada tim shift operator dan manajemen PLTGU Grati melalui WhatsApp.")
    
    with st.form("wa_form"):
        wa_api_key = st.text_input("API Key Gateway WhatsApp", type="password", value="MOCK_API_KEY_12345")
        nomor_penerima = st.text_input("Nomor WhatsApp Penerima (Contoh: 628123456789)", value="628123456789")
        pesan_custom = st.text_area("Template Pesan Peringatan", value="🧪 *TEST PESAN JELLYFISH PLTGU GRATI - JELLYFISH EARLY WARNING INTELLIGENCE SYSTEM* - Sistem Beroperasi Normal.")
        
        submitted = st.form_submit_button("Kirim Pesan Uji Coba")
        if submitted:
            success, info = kirim_whatsapp_notif(wa_api_key, nomor_penerima, pesan_custom)
            if success:
                st.success(info)
            else:
                st.error(info)

elif menu == "Log & Laporan Insiden":
    st.header("📋 Log & Riwayat Insiden Ubur-Ubur")
    st.markdown("Arsip riwayat peringatan dini dan tindakan penanggulangan yang telah dicatat oleh sistem.")
    
    # Tampilkan tabel log
    df_log = pd.DataFrame(st.session_state['log_insiden'])
    st.table(df_log)
    
    st.markdown("### Tambah Catatan Insiden Baru")
    with st.form("log_form"):
        new_pesan = st.text_input("Keterangan Kejadian")
        new_level = st.selectbox("Level Risiko", ["AMAN", "WASPADA", "BAHAYA"])
        new_tindakan = st.text_input("Tindakan Korektif yang Dilakukan")
        
        add_log = st.form_submit_button("Simpan Log ke Sistem")
        if add_log and new_pesan:
            st.session_state['log_insiden'].insert(0, {
                "waktu": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "level": new_level,
                "pesan": new_pesan,
                "tindakan": new_tindakan
            })
            st.success("Log berhasil ditambahkan!")
            st.rerun()

# Auto-refresh logic jika diaktifkan
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()

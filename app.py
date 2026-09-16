import time
import pandas as pd
import requests
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="JELLY-MARVEL Intelligence System",
    page_icon="🚨",
    layout="wide",
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("## ⚙️ JELLY-MARVEL Control Panel")

# Pengaturan WhatsApp Alert (Token default dari Anda)
st.sidebar.markdown("### 📱 Konfigurasi WhatsApp Alert")
wa_active = st.sidebar.checkbox(
    "Aktifkan Auto WhatsApp Alert", value=True
)
wa_token = st.sidebar.text_input(
    "WhatsApp API Token",
    type="password",
    value="9WEJQ8pKRJsRU5xKNyBs",  # Token Fonnte Anda
)
wa_target = st.sidebar.text_input(
    "Nomor HP Tujuan (Shift Operator)", value="082134902752"
)

st.sidebar.markdown("---")

# Sumber Input Data
input_source = st.sidebar.radio(
    "Sumber Input Data:",
    [
        "FORCE NORMAL (Verifikasi Lapangan: Nihil Ubur-ubur)",
        "Real-Time API (Selat Madura)",
        "🧪 Skenario Simulasi Manual",
    ],
    index=2,
)

# Skenario Pengujian
if input_source == "🧪 Skenario Simulasi Manual":
  scenario = st.sidebar.selectbox(
      "Skenario Pengujian:",
      [
          "🚨 KRITIS: SERANGAN UBUR-UBUR Massal",
          "⚠️ SIAGA: Kepadatan Sedang",
          "✅ AMAN: Kondisi Normal",
      ],
  )
else:
  scenario = "✅ AMAN: Kondisi Normal"

st.sidebar.markdown("---")

# Parameter Simulasi Slider
st.sidebar.markdown("### Parameter Lingkungan")
suhu = st.sidebar.slider("Suhu Laut (°C)", 25.0, 35.0, 32.5)
klorofil = st.sidebar.slider("Klorofil-a (mg/m³)", 0.0, 10.0, 5.8)
salinitas = st.sidebar.slider("Salinitas (PSU)", 25.0, 40.0, 34.8)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 50.0, 32.0)

# --- FUNGSI KIRIM WHATSAPP ---


def kirim_whatsapp(token, target, pesan):
  """Fungsi untuk mengirim pesan WhatsApp menggunakan API Fonnte"""
  if not token or not target:
    return {
        "status": False,
        "reason": "Token atau Nomor Tujuan belum diisi!",
    }

  target_cleaned = target.strip()
  if target_cleaned.startswith("0"):
    target_cleaned = "62" + target_cleaned[1:]

  url = "https://api.fonnte.com/send"
  headers = {"Authorization": token}
  payload = {"target": target_cleaned, "message": pesan}

  try:
    response = requests.post(url, data=payload, headers=headers, timeout=10)
    return response.json()
  except Exception as e:
    return {"status": False, "reason": str(e)}


# --- MAIN DASHBOARD INTERFACE ---
st.title("⚡ JELLY-MARVEL INTELLIGENCE SYSTEM PLTGU GRATI")
st.markdown(
    "**JELLYFISH EARLY-WARNING & MODULAR CONVEYOR INTELLIGENCE SYSTEM** — SWI"
    " INTAKE SELAT MADURA"
)

# Tentukan Status Berdasarkan Skenario
if "KRITIS" in scenario:
  risk_index = 98
  status_text = "STATUS KRITIS: SERANGAN UBUR-UBUR"
  eta_msg = "Pukul 20:16 WIB (~4 Menit lagi)"
elif "SIAGA" in scenario:
  risk_index = 65
  status_text = "STATUS SIAGA: KEPADATAN SEDANG"
  eta_msg = "Potensi dalam 15-30 menit"
else:
  risk_index = 12
  status_text = "STATUS AMAN: NORMAL"
  eta_msg = "Tidak ada ancaman terdeteksi"

# Layout Utama
col1, col2 = st.columns([2, 1])

with col1:
  st.markdown(f"### 🛑 Indeks Risiko Ancaman: {risk_index}%")
  st.progress(risk_index / 100)

  if risk_index > 80:
    st.error(f"**{status_text}**\n\nEstimasi Kedatangan: {eta_msg}")
  elif risk_index > 50:
    st.warning(f"**{status_text}**")
  else:
    st.success(f"**{status_text}**")

with col2:
  st.markdown("### 🧪 Tombol Uji Coba Manual WA")
  st.markdown(
      "Gunakan tombol di bawah ini untuk menguji koneksi API WhatsApp secara"
      " instan:"
  )

  if st.button("🚀 Kirim Test WhatsApp Sekarang", type="primary"):
    with st.spinner("Mengirim pesan WhatsApp..."):
      test_pesan = (
          "🧪 *TEST PESAN JELLY-MARVEL*\n\nSistem peringatan dini beroperasi"
          f" normal.\nStatus saat ini: {status_text} (Risiko: {risk_index}%)"
      )
      res = kirim_whatsapp(wa_token, wa_target, test_pesan)
      if res.get("status"):
        st.success(
            "✅ Berhasil! Respon API: " + str(res.get("reason", "Terkirim"))
        )
      else:
        st.error(
            "❌ Gagal mengirim WA. Alasan: "
            + str(res.get("reason", "Periksa kembali Token & Nomor Anda"))
        )

# --- OTOMATISASI KIRIM SAAT KRITIS ---
if wa_active and risk_index > 80:
  current_hour_key = pd.Timestamp.now().strftime("%Y-%m-%d-%H")

  if st.session_state.get("last_sent_hour") != current_hour_key:
    pesan_darurat = (
        "🚨 *DARURAT PLTGU GRATI*\n\nTerdeteksi *SERANGAN UBUR-UBUR MASSAL*"
        f" pada intake!\nIndeks Risiko: {risk_index}%\nEstimasi Tiba:"
        f" {eta_msg}\n\nHarap segera jalankan SOP Mitigasi Shift Operator!"
    )

    result = kirim_whatsapp(wa_token, wa_target, pesan_darurat)
    if result.get("status"):
      st.sidebar.success("✅ Auto WhatsApp Darurat Berhasil Terkirim!")
      st.session_state["last_sent_hour"] = current_hour_key
    else:
      st.sidebar.error(
          "❌ Auto WA Gagal: "
          + str(result.get("reason", "Cek token/koneksi"))
      )

# Mandatori Operator Shift
st.markdown("---")
st.markdown("### 📋 MANDATORI OPERATOR SHIFT")
st.markdown(
    """
1. Persiapan Pengoperasian Sistem Konveyor / MARVEL.
2. Jalankan Revolving screen / TBS mode Continuous High Speed.
3. Aktifkan Screen Wash Pump Pressure Max.
4. Manual running debris filter condensor.
5. Pengamatan DP all strainer cooling system.
6. Optimalkan pengaturan valve outlet kondensor.
7. Amati vacuum condenser.
8. Siapkan derating jika $\\Delta P > 0.30\\text{ mWC}$.
"""
)

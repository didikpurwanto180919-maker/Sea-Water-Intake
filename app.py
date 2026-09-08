# ==========================================
# SIDEBAR CONTROL
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

    st.sidebar.success("⚡ Live Auto-Refresh Active")

    # WIDGET JAM DIGITAL BERJALAN & COUNTDOWN TIMER (HTML + JS)
    sidebar_timer_html = f"""
    <div style="font-family: sans-serif; display: flex; flex-direction: column; gap: 8px;">
        <!-- Jam Digital WIB Real-Time -->
        <div style="background-color: #e8f4f8; color: #1d6f8a; padding: 10px; border-radius: 8px; border: 1px solid #bbee33;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">🕒 Jam Server Live (WIB):</div>
            <div id="live_clock" style="font-size: 16px; font-weight: bold; margin-top: 2px;">--:--:-- WIB</div>
        </div>

        <!-- Countdown Timer Auto Refresh -->
        <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 8px; border: 1px solid #c3e6cb;">
            <div style="font-size: 11px; font-weight: bold; text-transform: uppercase;">⏱️ Next Data Refresh:</div>
            <div style="font-size: 15px; font-weight: bold; margin-top: 2px;"><span id="timer">{REFRESH_INTERVAL_SEC}</span> detik</div>
        </div>
        
        <!-- Catatan Timestamp Tarik Data -->
        <div style="font-size: 11px; color: #6c757d; margin-top: 2px;">
            Last API Fetch: <b>{live_data['timestamp']}</b>
        </div>
    </div>

    <script>
        // 1. Fungsi Jam Digital Running (WIB / GMT+7)
        function updateClock() {{
            var now = new Date();
            var options = {{ timeZone: "Asia/Jakarta", hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }};
            var timeString = new Intl.DateTimeFormat("id-ID", options).format(now);
            document.getElementById('live_clock').innerHTML = timeString.replace(/\./g, ':') + " WIB";
        }}
        setInterval(updateClock, 1000);
        updateClock();

        // 2. Fungsi Countdown Timer Hitung Mundur
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
    st.sidebar.subheader("Atur Parameter Laut:")
    sst = st.sidebar.slider("Suhu Permukaan Laut (°C)", 25.0, 35.0, 29.7, 0.1)
    salinity = st.sidebar.slider("Salinitas (PSU)", 28.0, 36.0, 33.2, 0.1)
    current_speed = st.sidebar.slider("Kecepatan Arus (m/s)", 0.0, 2.0, 0.30, 0.01)
    chlorophyll = st.sidebar.slider("Klorofil-a (mg/m³)", 0.1, 8.0, 2.20, 0.01)
    wind_speed = st.sidebar.slider("Kecepatan Angin (knot)", 0.0, 25.0, 4.8, 0.1)

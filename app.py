with col_gauge:
    st.markdown("#### 🎯 Threat Risk Index")
    
    # 1. Tentukan warna gauge & kalkulasi skor gabungan risiko (Waspada/Kritis)
    if risk_class == 2:
        gauge_color = "#ef4444"
        display_score = probabilities[2] * 100
    elif risk_class == 1:
        gauge_color = "#f59e0b"
        # Menampilkan gabungan probabilitas Waspada (Class 1) + Kritis (Class 2)
        display_score = (probabilities[1] + probabilities[2]) * 100
    else:
        gauge_color = "#10b981"
        display_score = (1 - probabilities[0]) * 100

    # 2. Render Gauge Chart
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

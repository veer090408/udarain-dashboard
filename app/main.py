import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="UdaRain - Asisten Aktivitas Outdoor Depok",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pemetaan Lokasi Depok (Koordinat Open-Meteo & Kode Wilayah adm4 BMKG)
LOKASI_DEPOK = {
    "Pondok Cina (Kampus UI)": {"lat": -6.3646, "lon": 106.8286, "adm4": "32.76.06.1003"},
    "Kukusan": {"lat": -6.3683, "lon": 106.8167, "adm4": "32.76.06.1002"},
    "Beji": {"lat": -6.3750, "lon": 106.8194, "adm4": "32.76.06.1001"},
    "Kemiri Muka (Margonda)": {"lat": -6.3789, "lon": 106.8333, "adm4": "32.76.06.1004"},
    "Pancoran Mas": {"lat": -6.3958, "lon": 106.8181, "adm4": "32.76.01.1001"},
}

# ==========================================
# 2. DUMMY DATA (SEMENTARA SEBELUM INTEGRASI)
# ==========================================
def get_dummy_data(persona: str):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Data Historis (24 jam terakhir)
    hist_times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    df_hist = pd.DataFrame({
        "timestamp": hist_times,
        "pm25": np.random.uniform(25, 65, size=24).round(1),
        "rain_prob": np.random.uniform(5, 40, size=24).round(1),
        "temp": np.random.uniform(25, 32, size=24).round(1)
    })

    # Data Prediksi 6 jam ke depan (Sesuai kontrak src/models.py milik Veer)
    pred_times = [now + timedelta(hours=i) for i in range(1, 7)]
    df_pred = pd.DataFrame({
        "timestamp": pred_times,
        "pred_pm25": [38.2, 41.5, 45.0, 52.1, 48.0, 35.4],
        "pred_aqi": [108, 116, 124, 142, 132, 100],
        "pred_rain_prob": [15.0, 20.0, 65.0, 80.0, 45.0, 10.0],
        "pred_temp": [31.2, 30.5, 28.0, 26.8, 27.2, 27.5]
    })

    # Simulasi output scoring.py milik Rixq
    if persona == "Pelari":
        status, score, alasan = "Sedang", 68, "PM2.5 mulai meningkat (38.2 µg/m³), kurangi intensitas lari jarak jauh."
        best_hour = (now + timedelta(hours=6)).strftime("%H:%M WIB")
    elif persona == "Pengendara Motor":
        status, score, alasan = "Baik", 85, "Jalanan kering saat ini, namun waspadai potensi hujan tinggi 3 jam lagi."
        best_hour = (now + timedelta(hours=1)).strftime("%H:%M WIB")
    else: # Pejalan Kaki
        status, score, alasan = "Baik", 80, "Cuaca cukup bersahabat untuk berjalan kaki di luar ruangan."
        best_hour = (now + timedelta(hours=1)).strftime("%H:%M WIB")

    # Simulasi output bmkg_loader.py milik Rixq
    df_bmkg = pd.DataFrame({
        "timestamp": [now + timedelta(hours=3), now + timedelta(hours=6)],
        "cuaca_desc": ["Hujan Ringan", "Berawan"],
        "suhu": [28, 27],
        "kelembapan": [80, 85],
        "kecepatan_angin": [10, 8]
    })

    return df_hist, df_pred, status, score, alasan, best_hour, df_bmkg

# ==========================================
# 3. SIDEBAR (KONTROL USER)
# ==========================================
with st.sidebar:
    st.title("🌦️ UdaRain")
    st.caption("Asisten Aktivitas Outdoor Berbasis Prediksi Kualitas Udara & Hujan")
    st.divider()

    selected_loc = st.selectbox("📍 Pilih Lokasi (Depok)", list(LOKASI_DEPOK.keys()))
    loc_info = LOKASI_DEPOK[selected_loc]

    selected_persona = st.radio(
        "👤 Pilih Persona Aktivitas",
        ["Pelari", "Pengendara Motor", "Pejalan Kaki"],
        help="Bobot penilaian menyesuaikan sensitivitas aktivitas terhadap polusi dan hujan."
    )

    st.divider()
    if st.button("🔄 Perbarui Data", use_container_width=True):
        st.toast("Data berhasil diperbarui!", icon="✅")

    st.caption(f"Koordinat: `{loc_info['lat']}, {loc_info['lon']}`")
    st.caption(f"Kode BMKG: `{loc_info['adm4']}`")

# ==========================================
# 4. KONTEN UTAMA DASHBOARD
# ==========================================
df_hist, df_pred, status, score, alasan, best_hour, df_bmkg = get_dummy_data(selected_persona)

st.subheader(f"Kondisi & Rekomendasi di {selected_loc}")

# --- A. BANNER REKOMENDASI UTAMA ---
banner_color = {
    "Baik": ("#d4edda", "#155724", "✅"),
    "Sedang": ("#fff3cd", "#856404", "⚠️"),
    "Hindari": ("#f8d7da", "#721c24", "🛑")
}
bg_col, text_col, icon = banner_color[status]

st.markdown(
    f"""
    <div style="background-color: {bg_col}; color: {text_col}; padding: 20px; border-radius: 10px; border-left: 6px solid {text_col}; margin-bottom: 20px;">
        <h3 style="margin:0; color: {text_col};">{icon} Status Kelayakan ({selected_persona}): {status.upper()} (Skor: {score}/100)</h3>
        <p style="margin: 8px 0 4px 0; font-size: 16px;">{alasan}</p>
        <p style="margin: 0; font-size: 14px;"><b>🕒 Rekomendasi Jam Terbaik (6 Jam ke Depan):</b> Pukul {best_hour}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- B. 4 KARTU METRIK KONDISI TERKINI ---
col1, col2, col3, col4 = st.columns(4)
current_temp = df_hist["temp"].iloc[-1]
current_pm25 = df_hist["pm25"].iloc[-1]
next_rain = df_pred["pred_rain_prob"].iloc[0]
next_aqi = df_pred["pred_aqi"].iloc[0]

col1.metric("🌡️ Suhu Saat Ini", f"{current_temp} °C", f"{(df_pred['pred_temp'].iloc[0] - current_temp):.1f} °C (1 jam lagi)")
col2.metric("🌧️ Peluang Hujan (1 Jam)", f"{next_rain}%", "Waspada" if next_rain > 50 else "Aman", delta_color="inverse")
col3.metric("😷 PM2.5 Saat Ini", f"{current_pm25} µg/m³", f"{(df_pred['pred_pm25'].iloc[0] - current_pm25):.1f} µg/m³", delta_color="inverse")
col4.metric("📊 Estimasi US AQI", f"{next_aqi}", "Sedang" if next_aqi > 100 else "Baik", delta_color="off")

st.divider()

# --- C. GRAFIK TREN HISTORIS & PREDIKSI (PLOTLY) ---
st.subheader("📈 Tren Historis (24 Jam) & Prediksi (6 Jam ke Depan)")
tab_pm25, tab_rain = st.tabs(["Kualitas Udara (PM2.5)", "Peluang Hujan (%)"])

with tab_pm25:
    fig_pm = go.Figure()
    fig_pm.add_trace(go.Scatter(
        x=df_hist["timestamp"], y=df_hist["pm25"],
        mode="lines+markers", name="Historis (Open-Meteo)",
        line=dict(color="#1f77b4", width=2)
    ))
    x_pred_conn = [df_hist["timestamp"].iloc[-1]] + list(df_pred["timestamp"])
    y_pred_conn = [df_hist["pm25"].iloc[-1]] + list(df_pred["pred_pm25"])
    fig_pm.add_trace(go.Scatter(
        x=x_pred_conn, y=y_pred_conn,
        mode="lines+markers", name="Prediksi Model ML",
        line=dict(color="#ff7f0e", width=2, dash="dash")
    ))
    fig_pm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig_pm, use_container_width=True)

with tab_rain:
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Scatter(
        x=df_hist["timestamp"], y=df_hist["rain_prob"],
        mode="lines+markers", name="Historis",
        line=dict(color="#2ca02c", width=2)
    ))
    x_rain_conn = [df_hist["timestamp"].iloc[-1]] + list(df_pred["timestamp"])
    y_rain_conn = [df_hist["rain_prob"].iloc[-1]] + list(df_pred["pred_rain_prob"])
    fig_rain.add_trace(go.Scatter(
        x=x_rain_conn, y=y_rain_conn,
        mode="lines+markers", name="Prediksi Peluang Hujan",
        line=dict(color="#d62728", width=2, dash="dash")
    ))
    fig_rain.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Peluang Hujan (%)")
    st.plotly_chart(fig_rain, use_container_width=True)

st.divider()

# --- D. KOMPARASI PREDIKSI MODEL VS RESMI BMKG ---
st.subheader("⚖️ Komparasi Prediksi UdaRain vs Prakiraan Resmi BMKG")
col_ml, col_bmkg = st.columns(2)

with col_ml:
    st.markdown("**🤖 Prediksi Model UdaRain (Per Jam)**")
    df_pred_display = df_pred[["timestamp", "pred_temp", "pred_rain_prob", "pred_pm25"]].copy()
    df_pred_display["timestamp"] = df_pred_display["timestamp"].dt.strftime("%H:%M")
    df_pred_display.columns = ["Jam", "Suhu (°C)", "Peluang Hujan (%)", "PM2.5 (µg/m³)"]
    st.dataframe(df_pred_display, use_container_width=True, hide_index=True)

with col_bmkg:
    st.markdown("**🏛️ Prakiraan Resmi BMKG (Per 3 Jam)**")
    df_bmkg_display = df_bmkg.copy()
    df_bmkg_display["timestamp"] = df_bmkg_display["timestamp"].dt.strftime("%H:%M")
    df_bmkg_display.columns = ["Jam", "Cuaca", "Suhu (°C)", "Kelembapan (%)", "Angin (km/j)"]
    st.dataframe(df_bmkg_display, use_container_width=True, hide_index=True)

# --- E. FOOTER & DISCLAIMER WAJIB ---
st.divider()
st.caption("📚 **Sumber Data:** Open-Meteo (CC BY 4.0) & Badan Meteorologi, Klimatologi, dan Geofisika (BMKG).")
st.info(
    "⚠️ **Disclaimer:** Prediksi yang dihasilkan oleh model Machine Learning pada aplikasi ini bersifat eksperimental "
    "dan bukan pengganti peringatan dini cuaca resmi dari pemerintah."
)
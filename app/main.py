import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.bmkg_loader import DEPOK_LOCATIONS, fetch_bmkg_with_status
from src.data_loader import fetch_open_meteo_current
from src.models import predict_next_hours
from src.scoring import VALID_PERSONAS, calculate_activity_score


st.set_page_config(
    page_title="UdaRain | Aktivitas Outdoor Depok",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Baik": {"accent": "#047857", "start": "#ECFDF5", "end": "#D1FAE5", "border": "#6EE7B7"},
    "Sedang": {"accent": "#B45309", "start": "#FFFBEB", "end": "#FEF3C7", "border": "#FCD34D"},
    "Hindari": {"accent": "#B91C1C", "start": "#FEF2F2", "end": "#FEE2E2", "border": "#FCA5A5"},
}


@st.cache_data(ttl=1800, show_spinner=False)
def load_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    return fetch_open_meteo_current(lat=lat, lon=lon, past_days=2, forecast_days=1)


@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions(lat: float, lon: float, hours: int) -> pd.DataFrame:
    return predict_next_hours(lat=lat, lon=lon, hours=hours)


@st.cache_data(ttl=1800, show_spinner=False)
def load_bmkg(adm4_code: str) -> tuple[pd.DataFrame, str | None]:
    return fetch_bmkg_with_status(adm4_code=adm4_code)


def fmt(value, digits: int = 1, fallback: str = "—") -> str:
    try:
        if pd.isna(value):
            return fallback
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def status_color(status: str) -> dict[str, str]:
    return COLORS.get(status, COLORS["Sedang"])


def make_trend_chart(df_hist: pd.DataFrame, df_pred: pd.DataFrame, column: str, pred_column: str, title: str, unit: str, color: str) -> go.Figure:
    fig = go.Figure()
    if not df_hist.empty and column in df_hist:
        hist = df_hist.dropna(subset=[column])
        fig.add_trace(go.Scatter(
            x=hist["time"], y=hist[column], mode="lines", name="Historis",
            line={"color": color, "width": 2.5},
            fill="tozeroy", fillcolor=f"rgba(37, 99, 235, 0.08)",
            hovertemplate=f"%{{x|%d %b, %H:%M}} WIB<br>{title}: %{{y:.1f}} {unit}<extra></extra>",
        ))
    if not df_pred.empty and pred_column in df_pred:
        pred = df_pred.dropna(subset=[pred_column])
        if not df_hist.empty and column in df_hist and not df_hist[column].dropna().empty:
            last = df_hist.dropna(subset=[column]).iloc[-1]
            pred_x = [last["time"], *pred["timestamp"].tolist()]
            pred_y = [last[column], *pred[pred_column].tolist()]
        else:
            pred_x, pred_y = pred["timestamp"].tolist(), pred[pred_column].tolist()
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y, mode="lines+markers", name="Prediksi UdaRain",
            line={"color": "#7C3AED", "width": 2.5, "dash": "dot"},
            marker={"size": 6},
            hovertemplate=f"%{{x|%d %b, %H:%M}} WIB<br>Prediksi: %{{y:.1f}} {unit}<extra></extra>",
        ))
    fig.update_layout(
        title={"text": title, "font": {"size": 15}}, height=330,
        margin={"l": 8, "r": 12, "t": 44, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#475569"},
        hovermode="x unified", legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"showgrid": False, "automargin": True},
        yaxis={"title": unit, "gridcolor": "#E2E8F0", "zeroline": False, "automargin": True},
    )
    return fig


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root { --ink:#0F172A; --muted:#475569; --line:#E2E8F0; --canvas:#F4F7FB; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--ink); }
.stApp { background:var(--canvas); }
[data-testid="stHeader"] { background:rgba(244,247,251,.88); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
.block-container { max-width:1440px; padding:2rem 2.4rem 3rem; }
h1,h2,h3,h4,.metric-value { font-family:'Plus Jakarta Sans',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
.eyebrow { color:#64748B; font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.page-title { font:800 clamp(1.8rem,3vw,2.5rem)/1.15 'Plus Jakarta Sans',sans-serif; margin:.25rem 0 .35rem; }
.page-subtitle { color:#475569; margin:0 0 1.35rem; }
.live-pill { display:inline-flex; align-items:center; gap:.5rem; border:1px solid #BBF7D0; background:#F0FDF4; color:#047857; border-radius:999px; padding:.4rem .75rem; font-size:.78rem; font-weight:700; white-space:nowrap; }
.live-dot { width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 0 0 rgba(16,185,129,.5); animation:pulse 1.8s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 7px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@keyframes fadeInUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.hero-card,.metric-card,.surface-card { background:#fff; border:1px solid var(--line); border-radius:20px; box-shadow:0 4px 16px rgba(15,23,42,.035); transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; animation:fadeInUp .55s cubic-bezier(.16,1,.3,1) both; }
.hero-card:hover,.metric-card:hover,.surface-card:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(15,23,42,.09); border-color:#CBD5E1; }
.hero-card { padding:1.55rem 1.65rem; min-height:306px; }
.hero-top { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.persona-badge { border:1px solid #DBEAFE; background:#EFF6FF; color:#1D4ED8; padding:.38rem .7rem; border-radius:999px; font-size:.76rem; font-weight:700; }
.status-badge { padding:.38rem .7rem; border-radius:999px; font-size:.76rem; font-weight:800; letter-spacing:.05em; }
.hero-score { font:800 clamp(2.2rem,5vw,3.2rem)/1.08 'Plus Jakarta Sans',sans-serif; margin:1.15rem 0 .2rem; }
.hero-score small { font:600 1rem Inter,sans-serif; color:#64748B; letter-spacing:0; }
.score-track { width:100%; height:8px; border-radius:999px; background:#E2E8F0; overflow:hidden; margin:.8rem 0 1rem; }
.score-fill { height:100%; border-radius:999px; }
.hero-reason { color:#334155; line-height:1.65; margin:.9rem 0 1.1rem; }
.best-hour { border-radius:14px; padding:.9rem 1rem; display:flex; justify-content:space-between; gap:1rem; align-items:center; }
.best-label { font-size:.74rem; font-weight:700; opacity:.8; text-transform:uppercase; letter-spacing:.08em; }
.best-time { font:700 1.22rem 'Plus Jakarta Sans',sans-serif; margin-top:.15rem; }
.best-score { font-weight:800; white-space:nowrap; }
.metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.metric-card { padding:1.15rem; min-height:145px; }
.metric-label { color:#64748B; font-size:.78rem; font-weight:600; }
.metric-head { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.metric-icon { display:inline-flex; color:#64748B; }
.metric-icon svg { width:19px; height:19px; stroke:currentColor; fill:none; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.metric-value { font-size:clamp(1.55rem,3vw,2rem); font-weight:800; margin:.6rem 0 .35rem; }
.metric-foot { color:#64748B; font-size:.76rem; }
.metric-accent { width:30px; height:3px; border-radius:9px; margin-top:.75rem; }
.section-heading { font:700 1.15rem 'Plus Jakarta Sans',sans-serif; margin:1.6rem 0 .3rem; }
div[data-testid="stTabs"] button { color:#64748B; font-weight:600; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#0F172A; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
@media(max-width:768px) {
 .block-container { padding:.9rem .85rem 2rem; }
 .hero-card { padding:1.1rem; min-height:0; }
 .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
 .metric-card { padding:.9rem; min-height:130px; }
 .metric-value { font-size:clamp(1.35rem,7vw,1.85rem); }
 .hero-score { font-size:clamp(2rem,10vw,3rem); }
 .best-hour { align-items:flex-start; flex-direction:column; gap:.5rem; }
 [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.75rem; }
 [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:100% !important; }
 [data-testid="stDataFrame"] { overflow-x:auto; }
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div class='eyebrow'>UDA RAIN</div>", unsafe_allow_html=True)
    st.title("UdaRain")
    st.caption("Asisten aktivitas outdoor berbasis cuaca dan kualitas udara.")
    st.divider()
    selected_loc = st.selectbox("Lokasi di Depok", list(DEPOK_LOCATIONS.keys()))
    loc_info = DEPOK_LOCATIONS[selected_loc]
    selected_persona = st.selectbox("Aktivitas", list(VALID_PERSONAS), index=0)
    horizon_label = st.radio("Horizon prediksi", ["6 Jam", "12 Jam"], horizontal=True)
    horizon_hours = 6 if horizon_label == "6 Jam" else 12
    st.divider()
    if st.button("Perbarui Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Koordinat: {loc_info['lat']}, {loc_info['lon']}")
    st.caption(f"Kode BMKG: {loc_info['adm4']}")


now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
df_raw = pd.DataFrame()
df_pred = pd.DataFrame()
df_bmkg = pd.DataFrame()
warnings = []

try:
    with st.spinner("Mengambil kondisi cuaca dan kualitas udara…"):
        df_raw = load_open_meteo(float(loc_info["lat"]), float(loc_info["lon"]))
except Exception as exc:
    warnings.append(f"Data Open-Meteo belum tersedia ({type(exc).__name__}).")

try:
    with st.spinner("Menyiapkan prediksi model…"):
        df_pred = load_predictions(float(loc_info["lat"]), float(loc_info["lon"]), horizon_hours)
except Exception as exc:
    warnings.append(f"Prediksi model belum tersedia ({type(exc).__name__}). Pastikan model telah dilatih dan file model tersedia.")

try:
    df_bmkg, bmkg_warning = load_bmkg(loc_info["adm4"])
    if bmkg_warning:
        warnings.append(bmkg_warning)
except Exception as exc:
    warnings.append(f"Data BMKG belum tersedia ({type(exc).__name__}).")
    df_bmkg = pd.DataFrame()

if not df_raw.empty and "time" in df_raw:
    df_raw["time"] = pd.to_datetime(df_raw["time"], errors="coerce")
    df_hist = df_raw.loc[df_raw["time"].le(now)].tail(24).copy()
else:
    df_hist = pd.DataFrame()

if not df_pred.empty:
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
    try:
        df_scored, best_hour_info = calculate_activity_score(df_pred, persona=selected_persona)
    except (ValueError, KeyError) as exc:
        warnings.append(f"Skor aktivitas tidak dapat dihitung ({exc}).")
        df_scored, best_hour_info = pd.DataFrame(), {}
else:
    df_scored, best_hour_info = pd.DataFrame(), {}

for warning in warnings:
    st.markdown(f"<div style='border:1px solid #FCD34D;background:#FFFBEB;color:#92400E;padding:.75rem 1rem;border-radius:12px;margin:.4rem 0'>{warning}</div>", unsafe_allow_html=True)

st.markdown(f"<div class='eyebrow'>DEPOK · {now.strftime('%A, %d %B %Y')}</div><div class='page-title'>Kondisi luar ruang</div><p class='page-subtitle'>{selected_loc} · rekomendasi personal untuk {selected_persona.lower()}.</p>", unsafe_allow_html=True)

if df_scored.empty:
    st.info("Rekomendasi belum dapat ditampilkan. Periksa ketersediaan data dan berkas model, lalu perbarui data.")
else:
    current = df_scored.iloc[0]
    status = str(current["status"])
    score = int(current["score"])
    palette = status_color(status)
    best = best_hour_info or {}
    best_score = int(best.get("score", 0))
    best_time = best.get("jam_str", "Belum tersedia")
    best_status = str(best.get("status", status))
    best_palette = status_color(best_status)
    reason = str(current.get("alasan", "Rekomendasi belum tersedia."))
    hero_col, metrics_col = st.columns([1.15, 1], gap="medium")
    with hero_col:
        st.markdown(f"""
    <div class="hero-card" style="border-color:{palette['border']};background:linear-gradient(130deg,{palette['start']},{palette['end']});">
      <div class="hero-top"><span class="persona-badge">{selected_persona}</span><span class="status-badge" style="background:#fff;color:{palette['accent']};">{status.upper()}</span></div>
      <div class="hero-score" style="color:{palette['accent']};">{score}<small> / 100 · kelayakan jam berikutnya</small></div>
      <div class="score-track"><div class="score-fill" style="width:{score}%;background:{palette['accent']};"></div></div>
      <div class="hero-reason">{reason}</div>
      <div class="best-hour" style="background:rgba(255,255,255,.78);border:1px solid {best_palette['border']};color:{best_palette['accent']};">
        <div><div class="best-label">Jam terbaik · {horizon_label} ke depan</div><div class="best-time">{best_time}</div></div><div class="best-score">Skor {best_score}/100</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    hist_latest = df_hist.iloc[-1] if not df_hist.empty else pd.Series(dtype="object")
    temp_now = hist_latest.get("temperature_2m", current.get("pred_temp"))
    pm_now = hist_latest.get("pm2_5", current.get("pred_pm25"))
    next_rain = current.get("pred_rain_prob")
    next_aqi = current.get("pred_aqi")
    temp_delta = float(current.get("pred_temp", 0)) - float(temp_now) if pd.notna(temp_now) else None
    pm_delta = float(current.get("pred_pm25", 0)) - float(pm_now) if pd.notna(pm_now) else None
    rain_value = fmt(next_rain, 0)
    aqi_label = "Baik" if pd.notna(next_aqi) and float(next_aqi) <= 50 else "Sedang" if pd.notna(next_aqi) and float(next_aqi) <= 100 else "Tinggi"
    metrics = [
        ("Suhu udara", f"{fmt(temp_now)}°", f"{temp_delta:+.1f}° menuju jam berikutnya" if temp_delta is not None else "Data tren belum tersedia", "#F97316", "<svg viewBox='0 0 24 24'><path d='M14 14.8V5a3 3 0 0 0-6 0v9.8a5 5 0 1 0 6 0Z'/><path d='M11 12V5'/></svg>"),
        ("Peluang hujan · 1 jam", f"{rain_value}%", "Prediksi UdaRain", "#2563EB", "<svg viewBox='0 0 24 24'><path d='M12 3C9 7 6 10.2 6 14a6 6 0 0 0 12 0c0-3.8-3-7-6-11Z'/><path d='M9 15a3 3 0 0 0 3 3'/></svg>"),
        ("Konsentrasi PM2.5", f"{fmt(pm_now)}", f"{pm_delta:+.1f} µg/m³ menuju jam berikutnya" if pm_delta is not None else "µg/m³ · data tren belum tersedia", "#7C3AED", "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='2'/><path d='M5 8h3M16 8h3M4 12h3M17 12h3M5 16h3M16 16h3'/></svg>"),
        ("Estimasi US AQI", fmt(next_aqi, 0), aqi_label, "#0F766E", "<svg viewBox='0 0 24 24'><path d='M4 18a8 8 0 1 1 16 0'/><path d='m12 14 4-4'/><path d='M7 18h10'/></svg>"),
    ]
    cards = "".join(f"<div class='metric-card'><div class='metric-head'><div class='metric-label'>{label}</div><span class='metric-icon'>{icon}</span></div><div class='metric-value'>{value}</div><div class='metric-foot'>{foot}</div><div class='metric-accent' style='background:{accent}'></div></div>" for label, value, foot, accent, icon in metrics)
    with metrics_col:
        st.markdown(f"<div style='height:14px'></div><div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)

st.markdown("<div class='section-heading'>Jelajahi kondisi</div>", unsafe_allow_html=True)
tab_trend, tab_bmkg, tab_schedule = st.tabs(["Tren Historis & Prediksi", "Komparasi Resmi BMKG", "Jadwal Kelayakan Per Jam"])

with tab_trend:
    if df_hist.empty and df_pred.empty:
        st.caption("Grafik akan tampil setelah data Open-Meteo dan prediksi tersedia.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_trend_chart(df_hist, df_pred, "pm2_5", "pred_pm25", "PM2.5", "µg/m³", "#2563EB"), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        with c2:
            st.plotly_chart(make_trend_chart(df_hist, df_pred, "temperature_2m", "pred_temp", "Suhu udara", "°C", "#F97316"), use_container_width=True, config={"displayModeBar": False, "responsive": True})
        st.caption("Garis penuh menunjukkan data historis Open-Meteo; garis putus-putus menunjukkan prediksi model UdaRain.")

with tab_bmkg:
    left, right = st.columns(2)
    with left:
        st.markdown("**Prediksi UdaRain · per jam**")
        if df_pred.empty:
            st.caption("Prediksi model belum tersedia.")
        else:
            model_view = df_pred[["timestamp", "pred_temp", "pred_rain_prob", "pred_pm25"]].copy()
            model_view["timestamp"] = model_view["timestamp"].dt.strftime("%d %b, %H:%M")
            model_view.columns = ["Waktu", "Suhu (°C)", "Hujan (%)", "PM2.5 (µg/m³)"]
            st.dataframe(model_view, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Prakiraan resmi BMKG · per 3 jam**")
        if df_bmkg.empty:
            st.caption("Prakiraan BMKG belum tersedia untuk lokasi ini.")
        else:
            bmkg_view = df_bmkg.copy()
            bmkg_view["timestamp"] = pd.to_datetime(bmkg_view["timestamp"]).dt.strftime("%d %b, %H:%M")
            bmkg_view.columns = ["Waktu", "Cuaca", "Suhu (°C)", "Kelembapan (%)", "Angin (km/j)" ]
            st.dataframe(bmkg_view, use_container_width=True, hide_index=True)

with tab_schedule:
    if df_scored.empty:
        st.caption("Jadwal kelayakan akan tampil setelah prediksi model tersedia.")
    else:
        schedule = df_scored[["timestamp", "score", "status", "pred_pm25", "pred_rain_prob", "pred_temp", "alasan"]].copy()
        schedule["timestamp"] = schedule["timestamp"].dt.strftime("%A, %d %b · %H:%M")
        schedule.columns = ["Waktu", "Skor", "Status", "PM2.5 (µg/m³)", "Hujan (%)", "Suhu (°C)", "Rekomendasi"]
        st.dataframe(schedule, use_container_width=True, hide_index=True, column_config={"Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%d/100")})

st.divider()
st.markdown("<div class='live-pill'><span class='live-dot'></span> Data diperbarui dari layanan cuaca</div>", unsafe_allow_html=True)
st.caption("Sumber: Open-Meteo dan Badan Meteorologi, Klimatologi, dan Geofisika (BMKG). Prediksi model bersifat eksperimental dan tidak menggantikan peringatan resmi BMKG.")

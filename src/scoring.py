"""
Scoring Engine & Rekomendasi Aktivitas Outdoor Berbasis Persona.
Bagian dari proyek UdaRain — Kontributor: Rixq
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

VALID_PERSONAS = ("Pelari", "Pengendara Motor", "Pejalan Kaki")

# Bobot sensitivitas tiap persona (total bobot = 1.0)
PERSONA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Pelari": {
        "air": 0.55,   # Laju pernapasan meningkat -> sangat sensitif PM2.5
        "rain": 0.20,  # Hujan ringan masih bisa ditoleransi dibanding polusi berat
        "temp": 0.25,  # Suhu panas berisiko heat stress saat lari
    },
    "Pengendara Motor": {
        "air": 0.15,   # Toleransi lebih tinggi terhadap PM2.5 (bisa memakai masker/helm)
        "rain": 0.70,  # Sangat sensitif hujan -> jalan licin & jarak pandang menurun
        "temp": 0.15,  # Terbantu aliran angin saat berkendara
    },
    "Pejalan Kaki": {
        "air": 0.35,   # Sensitivitas seimbang terhadap polusi udara
        "rain": 0.40,  # Cukup sensitif terhadap hujan saat berjalan kaki
        "temp": 0.25,  # Kenyamanan termal saat berjalan di ruang terbuka
    },
}


def _score_pm25(pm25: float) -> float:
    """Mengonversi konsentrasi PM2.5 (µg/m³) menjadi sub-skor kenyamanan 0–100."""
    if pd.isna(pm25):
        return 70.0
    pm25 = max(0.0, float(pm25))
    if pm25 <= 15.0:
        return 100.0
    if pm25 <= 35.0:
        return 100.0 - ((pm25 - 15.0) / 20.0) * 30.0
    if pm25 <= 55.0:
        return 70.0 - ((pm25 - 35.0) / 20.0) * 30.0
    if pm25 <= 100.0:
        return 40.0 - ((pm25 - 55.0) / 45.0) * 30.0
    return max(0.0, 10.0 - (pm25 - 100.0) * 0.2)


def _score_rain(rain_prob_pct: float) -> float:
    """Mengonversi peluang hujan (0–100%) menjadi sub-skor keamanan 0–100."""
    if pd.isna(rain_prob_pct):
        return 80.0
    prob = float(np.clip(rain_prob_pct, 0.0, 100.0))
    if prob <= 20.0:
        return 100.0
    if prob <= 50.0:
        return 100.0 - ((prob - 20.0) / 30.0) * 45.0
    if prob <= 80.0:
        return 55.0 - ((prob - 50.0) / 30.0) * 40.0
    return max(0.0, 15.0 - ((prob - 80.0) / 20.0) * 15.0)


def _score_temp(temp_c: float) -> float:
    """Mengonversi suhu udara (°C) menjadi sub-skor kenyamanan tropis 0–100."""
    if pd.isna(temp_c):
        return 80.0
    temp = float(temp_c)
    if 24.0 <= temp <= 28.0:
        return 100.0
    if temp < 24.0:
        return max(40.0, 100.0 - (24.0 - temp) * 6.0)
    if temp <= 32.0:
        return 100.0 - ((temp - 28.0) / 4.0) * 40.0
    return max(0.0, 60.0 - (temp - 32.0) * 12.0)


def _determine_status(score: int) -> str:
    """Mengategorikan skor 0–100 menjadi Baik, Sedang, atau Hindari."""
    if score >= 70:
        return "Baik"
    if score >= 45:
        return "Sedang"
    return "Hindari"


def _build_reason(
    persona: str,
    pm25: float,
    rain_pct: float,
    temp: float,
    sub_scores: Dict[str, float],
    status: str,
) -> str:
    """Menyusun alasan singkat & praktis berdasarkan faktor risiko paling dominan."""
    if status == "Baik":
        return (
            f"Kondisi ideal untuk {persona.lower()} "
            f"(PM2.5 {pm25:.0f} µg/m³, peluang hujan {rain_pct:.0f}%, suhu {temp:.1f}°C)."
        )

    worst_factor = min(sub_scores, key=sub_scores.get)

    if worst_factor == "air":
        if persona == "Pelari":
            return f"PM2.5 tinggi ({pm25:.0f} µg/m³), kurangi intensitas atau tunda sesi lari."
        if persona == "Pengendara Motor":
            return f"Polusi udara meningkat (PM2.5 {pm25:.0f} µg/m³), gunakan masker & visor tertutup."
        return f"Kualitas udara kurang sehat (PM2.5 {pm25:.0f} µg/m³), batasi durasi di luar ruangan."

    if worst_factor == "rain":
        if persona == "Pengendara Motor":
            return f"Peluang hujan tinggi ({rain_pct:.0f}%), waspada jalan licin & siapkan jas hujan."
        if persona == "Pelari":
            return f"Berpotensi hujan ({rain_pct:.0f}%), pertimbangkan rute pendek atau latihan indoor."
        return f"Peluang hujan cukup tinggi ({rain_pct:.0f}%), bawa payung sebelum berangkat."

    if temp > 28.0:
        if persona == "Pelari":
            return f"Suhu cukup panas ({temp:.1f}°C), jaga hidrasi ekstra & hindari lari terlalu siang."
        return f"Cuaca terik ({temp:.1f}°C), gunakan pelindung matahari & perbanyak minum air."
    return f"Suhu cenderung sejuk/dingin ({temp:.1f}°C), gunakan pakaian yang sesuai."


def calculate_activity_score(
    df_pred: pd.DataFrame, persona: str = "Pejalan Kaki"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Menghitung skor kelayakan aktivitas outdoor (0–100) untuk setiap jam prediksi
    dan mengidentifikasi jam terbaik dalam rentang prediksi.

    Parameters
    ----------
    df_pred : pd.DataFrame
        DataFrame prediksi dari models.py dengan kolom minimal:
        ['timestamp', 'pred_pm25', 'pred_rain_prob', 'pred_temp'] (dan opsional 'pred_aqi').
    persona : str, default="Pejalan Kaki"
        Pilihan persona: "Pelari", "Pengendara Motor", atau "Pejalan Kaki".

    Returns
    -------
    tuple[pd.DataFrame, dict]
        1. df_scored: Salinan df_pred dengan tambahan kolom ['score', 'status', 'alasan'].
        2. best_hour_info: Dictionary berisi detail jam terbaik untuk ditampilkan di banner UI.
    """
    if persona not in PERSONA_WEIGHTS:
        raise ValueError(
            f"Persona '{persona}' tidak dikenali. Pilih salah satu dari: {VALID_PERSONAS}"
        )

    required_cols = {"timestamp", "pred_pm25", "pred_rain_prob", "pred_temp"}
    missing_cols = required_cols - set(df_pred.columns)
    if missing_cols:
        raise KeyError(f"Kolom wajib tidak ditemukan pada df_pred: {missing_cols}")

    df_out = df_pred.copy()
    if df_out.empty:
        df_out["score"] = pd.Series(dtype="int64")
        df_out["status"] = pd.Series(dtype="object")
        df_out["alasan"] = pd.Series(dtype="object")
        return df_out, {}

    weights = PERSONA_WEIGHTS[persona]

    rain_series = pd.to_numeric(df_out["pred_rain_prob"], errors="coerce").fillna(0.0)
    if rain_series.max() <= 1.0:
        rain_pct_series = rain_series * 100.0
    else:
        rain_pct_series = rain_series

    scores = []
    statuses = []
    reasons = []

    for idx, row in df_out.iterrows():
        pm25 = float(row["pred_pm25"])
        rain_pct = float(rain_pct_series.loc[idx])
        temp = float(row["pred_temp"])

        sub_scores = {
            "air": _score_pm25(pm25),
            "rain": _score_rain(rain_pct),
            "temp": _score_temp(temp),
        }

        weighted_score = (
            weights["air"] * sub_scores["air"]
            + weights["rain"] * sub_scores["rain"]
            + weights["temp"] * sub_scores["temp"]
        )

        # Guardrail keselamatan ekstrem agar kondisi berbahaya tidak masuk kategori "Baik"
        if persona == "Pelari" and pm25 > 65.0:
            weighted_score = min(weighted_score, 44.0)
        elif persona == "Pengendara Motor" and rain_pct > 75.0:
            weighted_score = min(weighted_score, 44.0)

        final_score = int(np.clip(round(weighted_score), 0, 100))
        status = _determine_status(final_score)
        reason = _build_reason(persona, pm25, rain_pct, temp, sub_scores, status)

        scores.append(final_score)
        statuses.append(status)
        reasons.append(reason)

    df_out["score"] = scores
    df_out["status"] = statuses
    df_out["alasan"] = reasons

    best_idx = df_out["score"].idxmax()
    best_row = df_out.loc[best_idx]
    best_ts = pd.to_datetime(best_row["timestamp"])

    best_hour_info: Dict[str, Any] = {
        "timestamp": best_ts,
        "jam_str": best_ts.strftime("%H:%M WIB") if pd.notna(best_ts) else "-",
        "score": int(best_row["score"]),
        "status": best_row["status"],
        "alasan": best_row["alasan"],
        "pred_pm25": float(best_row["pred_pm25"]),
        "pred_rain_prob": float(rain_pct_series.loc[best_idx]),
        "pred_temp": float(best_row["pred_temp"]),
    }

    return df_out, best_hour_info


if __name__ == "__main__":
    dummy_pred = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-17 09:00", periods=6, freq="h"),
            "pred_pm25": [22.0, 28.5, 45.0, 68.0, 52.0, 30.0],
            "pred_aqi": [72, 85, 124, 157, 141, 89],
            "pred_rain_prob": [0.10, 0.15, 0.35, 0.80, 0.65, 0.20],
            "pred_temp": [27.5, 29.0, 31.2, 30.5, 28.0, 26.8],
        }
    )

    for p in VALID_PERSONAS:
        df_res, best = calculate_activity_score(dummy_pred, persona=p)
        print(f"\n=== Persona: {p} ===")
        print(f"Jam Terbaik: {best['jam_str']} (Skor: {best['score']} - {best['status']})")
        print(df_res[["timestamp", "score", "status", "alasan"]])

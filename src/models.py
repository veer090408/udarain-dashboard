import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from data_loader import fetch_open_meteo_current, load_from_sqlite
from processing import create_features, get_feature_columns

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PM25_MODEL_PATH = MODELS_DIR / "air_quality_model.joblib"
RAIN_MODEL_PATH = MODELS_DIR / "rain_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def pm25_to_us_aqi(pm25: float) -> int:
    """Konversi standar EPA dari konsentrasi PM2.5 (µg/m³) ke skala US AQI."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    pm = max(0.0, min(500.4, float(pm25)))
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low
            return int(round(aqi))
    return 500


def train_and_evaluate() -> dict:
    """Melatih model regresi (PM2.5) & klasifikasi (Hujan), membandingkan dengan baseline, lalu menyimpannya."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_from_sqlite("historical_depok")
    df_feat = create_features(df_raw, is_training=True)
    feature_cols = get_feature_columns(df_feat)

    # Time-based split (80% data awal untuk latih, 20% data terbaru untuk uji — anti data leakage)
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train_pm, y_test_pm = train_df["target_pm25_next_1h"], test_df["target_pm25_next_1h"]
    y_train_rain, y_test_rain = train_df["target_rain_next_1h"], test_df["target_rain_next_1h"]

    # --- 1. MODEL REGRESI KUALITAS UDARA (PM2.5) ---
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_pm)
    pred_pm = rf_reg.predict(X_test)

    # Baseline Regresi: Persistence (asumsi PM2.5 jam depan == PM2.5 jam ini)
    baseline_pm = test_df["pm2_5"].values

    mae_ml = float(mean_absolute_error(y_test_pm, pred_pm))
    rmse_ml = float(np.sqrt(mean_squared_error(y_test_pm, pred_pm)))
    mae_base = float(mean_absolute_error(y_test_pm, baseline_pm))
    rmse_base = float(np.sqrt(mean_squared_error(y_test_pm, baseline_pm)))

    # --- 2. MODEL KLASIFIKASI PELUANG HUJAN ---
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_rain)
    pred_rain = rf_clf.predict(X_test)

    # Baseline Klasifikasi: Persistence (asumsi hujan jam depan == kondisi hujan jam ini)
    baseline_rain = (test_df["rain"].values > 0.1).astype(int)

    f1_ml = float(f1_score(y_test_rain, pred_rain))
    f1_base = float(f1_score(y_test_rain, baseline_rain))

    # Simpan Top 10 Feature Importance untuk ditampilkan di UI nanti
    importances = pd.Series(rf_reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = {k: round(float(v), 4) for k, v in importances.head(10).items()}

    metrics = {
        "pm25_mae_ml": round(mae_ml, 3),
        "pm25_mae_baseline": round(mae_base, 3),
        "pm25_rmse_ml": round(rmse_ml, 3),
        "pm25_rmse_baseline": round(rmse_base, 3),
        "rain_f1_ml": round(f1_ml, 3),
        "rain_f1_baseline": round(f1_base, 3),
        "top_features": top_features,
        "feature_cols": feature_cols,
    }

    joblib.dump(rf_reg, PM25_MODEL_PATH)
    joblib.dump(rf_clf, RAIN_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def predict_next_hours(lat: float = -6.3645, lon: float = 106.8287, hours: int = 12) -> pd.DataFrame:
    """
    Menghasilkan prediksi N jam ke depan secara iteratif (multi-horizon).
    Output sesuai kontrak integrasi UI: ['timestamp', 'pred_pm25', 'pred_aqi', 'pred_rain_prob', 'pred_temp']
    """
    rf_reg = joblib.load(PM25_MODEL_PATH)
    rf_clf = joblib.load(RAIN_MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    df_recent = fetch_open_meteo_current(lat=lat, lon=lon, past_days=2, forecast_days=2)
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).floor("h")

    # Ambil data sampai jam sekarang sebagai konteks historis awal
    history_df = df_recent[df_recent["time"] <= now].copy().reset_index(drop=True)
    future_weather = df_recent[df_recent["time"] > now].copy().reset_index(drop=True)

    results = []
    steps = min(hours, len(future_weather))

    for i in range(steps):
        feat_df = create_features(history_df, is_training=False)
        latest_row = feat_df.iloc[[-1]][feature_cols]

        pred_pm25 = float(max(0.0, rf_reg.predict(latest_row)[0]))
        pred_rain_prob = float(rf_clf.predict_proba(latest_row)[0][1] * 100)

        next_row = future_weather.iloc[[i]].copy()
        next_time = next_row["time"].values[0]
        pred_temp = float(next_row["temperature_2m"].values[0])

        results.append(
            {
                "timestamp": pd.to_datetime(next_time),
                "pred_pm25": round(pred_pm25, 1),
                "pred_aqi": pm25_to_us_aqi(pred_pm25),
                "pred_rain_prob": round(pred_rain_prob, 1),
                "pred_temp": round(pred_temp, 1),
            }
        )

        # Masukkan hasil prediksi ke history_df agar iterasi jam berikutnya memakai nilai terbaru
        next_row["pm2_5"] = pred_pm25
        next_row["us_aqi"] = pm25_to_us_aqi(pred_pm25)
        history_df = pd.concat([history_df, next_row], ignore_index=True)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Melatih model Random Forest untuk PM2.5 dan Hujan...")
    metrics = train_and_evaluate()
    print("\n=== HASIL EVALUASI MODEL VS BASELINE ===")
    print(f"PM2.5 MAE  -> ML: {metrics['pm25_mae_ml']} | Baseline: {metrics['pm25_mae_baseline']}")
    print(f"PM2.5 RMSE -> ML: {metrics['pm25_rmse_ml']} | Baseline: {metrics['pm25_rmse_baseline']}")
    print(f"Rain F1    -> ML: {metrics['rain_f1_ml']} | Baseline: {metrics['rain_f1_baseline']}")

    print("\nMenguji prediksi 6 jam ke depan untuk Depok...")
    df_pred = predict_next_hours(hours=6)
    print(df_pred)
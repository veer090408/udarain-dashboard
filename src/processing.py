import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Mengubah data mentah cuaca & kualitas udara per jam menjadi fitur siap latih / inferensi.
    """
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time").reset_index(drop=True)

    # 1. Fitur Waktu Siklikal (Pola jam sibuk & musim)
    hour = data["time"].dt.hour
    month = data["time"].dt.month
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    data["is_weekend"] = (data["time"].dt.dayofweek >= 5).astype(int)

    # 2. Fitur Lag (Kondisi 1, 2, 3, 6, 12, dan 24 jam yang lalu)
    lag_cols = ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "rain"]
    lags = [1, 2, 3, 6, 12, 24]
    for col in lag_cols:
        if col in data.columns:
            for lag in lags:
                data[f"{col}_lag_{lag}h"] = data[col].shift(lag)

    # 3. Fitur Rolling Statistics (Rata-rata & tren perubahan 3 jam & 6 jam terakhir)
    for col in ["pm2_5", "relative_humidity_2m", "temperature_2m"]:
        if col in data.columns:
            data[f"{col}_roll_mean_3h"] = data[col].rolling(window=3, min_periods=1).mean()
            data[f"{col}_roll_mean_6h"] = data[col].rolling(window=6, min_periods=1).mean()
            data[f"{col}_diff_1h"] = data[col].diff(1)

    # 4. Target Prediksi (Hanya dibuat saat mode training)
    if is_training:
        data["target_pm25_next_1h"] = data["pm2_5"].shift(-1)
        data["target_rain_next_1h"] = (data["rain"].shift(-1) > 0.1).astype(int)
        data = data.dropna().reset_index(drop=True)

    return data


def get_feature_columns(df: pd.DataFrame) -> list:
    """Mengembalikan daftar nama kolom fitur (mengecualikan timestamp dan kolom target)."""
    exclude_cols = {
        "time",
        "us_aqi",
        "target_pm25_next_1h",
        "target_rain_next_1h",
    }
    return [col for col in df.columns if col not in exclude_cols]


if __name__ == "__main__":
    from data_loader import load_from_sqlite

    df_raw = load_from_sqlite("historical_depok")
    df_feat = create_features(df_raw, is_training=True)
    feat_cols = get_feature_columns(df_feat)

    print(f"Jumlah baris awal : {len(df_raw)}")
    print(f"Jumlah baris bersih: {len(df_feat)}")
    print(f"Total fitur dibuat : {len(feat_cols)} fitur")
    print("Contoh 5 fitur pertama:", feat_cols[:5])
    print("Distribusi target hujan (0=Tidak, 1=Hujan):\n", df_feat["target_rain_next_1h"].value_counts(normalize=True))
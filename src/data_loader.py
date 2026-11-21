import sqlite3
from pathlib import Path
import pandas as pd
import requests

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "udarain_cache.db"


def fetch_open_meteo_history(
    lat: float = -6.3645,
    lon: float = 106.8287,
    start_date: str = "2023-01-01",
    end_date: str = "2026-11-12",
) -> pd.DataFrame:
    """Menarik dan menggabungkan data historis kualitas udara & cuaca per jam."""
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,us_aqi",
        "domains": "cams_global",
        "timezone": "Asia/Jakarta",
    }
    r_air = requests.get(air_url, params=air_params, timeout=60)
    r_air.raise_for_status()
    df_air = pd.DataFrame(r_air.json()["hourly"])

    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
        "timezone": "Asia/Jakarta",
    }
    r_weather = requests.get(weather_url, params=weather_params, timeout=60)
    r_weather.raise_for_status()
    df_weather = pd.DataFrame(r_weather.json()["hourly"])

    df = pd.merge(df_air, df_weather, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_open_meteo_current(
    lat: float = -6.3645,
    lon: float = 106.8287,
    past_days: int = 2,
    forecast_days: int = 1,
) -> pd.DataFrame:
    """Menarik data 48 jam terakhir + 24 jam ke depan untuk inferensi model & UI."""
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "hourly": "pm2_5,pm10,us_aqi",
        "domains": "cams_global",
        "timezone": "Asia/Jakarta",
    }
    r_air = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=air_params,
        timeout=30,
    )
    r_air.raise_for_status()
    df_air = pd.DataFrame(r_air.json()["hourly"])

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
        "timezone": "Asia/Jakarta",
    }
    r_weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=weather_params,
        timeout=30,
    )
    r_weather.raise_for_status()
    df_weather = pd.DataFrame(r_weather.json()["hourly"])

    df = pd.merge(df_air, df_weather, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    return df


def save_to_sqlite(df: pd.DataFrame, table_name: str = "historical_depok") -> None:
    """Menyimpan DataFrame ke database SQLite lokal."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def load_from_sqlite(table_name: str = "historical_depok") -> pd.DataFrame:
    """Membaca DataFrame dari database SQLite lokal."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    df["time"] = pd.to_datetime(df["time"])
    return df


if __name__ == "__main__":
    print("Mengunduh data historis Depok (2023 - 2026)...")
    df_hist = fetch_open_meteo_history()
    save_to_sqlite(df_hist, "historical_depok")
    print(f"Selesai! Total {len(df_hist)} baris tersimpan ke {DB_PATH}")
    print(df_hist.tail(3))
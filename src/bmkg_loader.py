"""
Modul penarik dan parser data prakiraan cuaca publik BMKG (tingkat kelurahan/adm4).
Bagian dari proyek UdaRain — Kontributor: Rixq
"""

import logging
from typing import Tuple, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BMKG_BASE_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
REQUIRED_COLUMNS = [
    "timestamp",
    "cuaca_desc",
    "suhu",
    "kelembapan",
    "kecepatan_angin",
]

# Referensi kode wilayah (adm4) & koordinat sekitar Kota Depok untuk digunakan di UI (Kevin)
DEPOK_LOCATIONS = {
    "Pondok Cina (UI Depok)": {
        "adm4": "32.76.06.1004",
        "lat": -6.3647,
        "lon": 106.8286,
    },
    "Kukusan (Beji)": {
        "adm4": "32.76.06.1005",
        "lat": -6.3694,
        "lon": 106.8161,
    },
    "Beji (Pusat Kecamatan)": {
        "adm4": "32.76.06.1001",
        "lat": -6.3775,
        "lon": 106.8194,
    },
    "Kemiri Muka (Margonda)": {
        "adm4": "32.76.06.1003",
        "lat": -6.3746,
        "lon": 106.8328,
    },
    "Pancoran Mas": {
        "adm4": "32.76.01.1001",
        "lat": -6.3944,
        "lon": 106.8169,
    },
}


def _empty_bmkg_dataframe() -> pd.DataFrame:
    """Mengembalikan DataFrame kosong dengan skema kolom standar BMKG."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def fetch_bmkg_with_status(
    adm4_code: str, timeout: int = 10
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Menarik data prakiraan cuaca BMKG beserta pesan status/peringatan untuk UI.

    Parameters
    ----------
    adm4_code : str
        Kode wilayah tingkat kelurahan/desa Kemendagri (contoh: '32.76.06.1004').
    timeout : int, default=10
        Batas waktu tunggu HTTP request dalam detik.

    Returns
    -------
    tuple[pd.DataFrame, Optional[str]]
        (DataFrame hasil parsing, pesan error/peringatan jika gagal atau None jika sukses)
    """
    if not adm4_code or not isinstance(adm4_code, str):
        return _empty_bmkg_dataframe(), "Kode wilayah adm4 BMKG tidak valid."

    headers = {
        "User-Agent": "UdaRain-Dashboard/1.0 (Academic-Project)",
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            BMKG_BASE_URL,
            params={"adm4": adm4_code.strip()},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        data_list = payload.get("data", [])
        if not data_list or not isinstance(data_list, list):
            return (
                _empty_bmkg_dataframe(),
                f"Data prakiraan BMKG kosong untuk kode wilayah {adm4_code}.",
            )

        # Struktur JSON BMKG: data[0]["cuaca"] adalah list of lists (per hari -> per 3 jam)
        cuaca_nested = data_list[0].get("cuaca", [])
        records = []

        for daily_block in cuaca_nested:
            if not isinstance(daily_block, list):
                continue
            for item in daily_block:
                if not isinstance(item, dict):
                    continue

                # Prioritaskan local_datetime (sudah dalam WIB / Asia/Jakarta)
                raw_time = item.get("local_datetime") or item.get("datetime")
                if not raw_time:
                    continue

                records.append(
                    {
                        "timestamp": raw_time,
                        "cuaca_desc": item.get("weather_desc", "Tidak Diketahui"),
                        "suhu": item.get("t"),
                        "kelembapan": item.get("hu"),
                        "kecepatan_angin": item.get("ws"),
                    }
                )

        if not records:
            return (
                _empty_bmkg_dataframe(),
                "Format jadwal cuaca BMKG tidak ditemukan dalam respons API.",
            )

        df = pd.DataFrame(records)

        # Konversi tipe data & standardisasi timestamp (WIB naive agar cocok dengan Open-Meteo)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = (
                df["timestamp"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
            )

        for numeric_col in ["suhu", "kelembapan", "kecepatan_angin"]:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

        df = (
            df.dropna(subset=["timestamp"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return df[REQUIRED_COLUMNS], None

    except requests.exceptions.Timeout:
        msg = "Koneksi ke server BMKG timeout. Menampilkan prediksi model internal saja."
        logger.warning(msg)
        return _empty_bmkg_dataframe(), msg
    except requests.exceptions.RequestException as exc:
        msg = f"Gagal menghubungi API BMKG ({exc.__class__.__name__})."
        logger.warning(msg)
        return _empty_bmkg_dataframe(), msg
    except (ValueError, KeyError, TypeError) as exc:
        msg = f"Gagal mem-parsing data JSON BMKG: {exc}"
        logger.error(msg)
        return _empty_bmkg_dataframe(), msg


def get_bmkg_forecast(adm4_code: str, timeout: int = 10) -> pd.DataFrame:
    """
    Kontrak fungsi utama sesuai Master Blueprint.
    Mengambil & mem-parsing prakiraan cuaca BMKG menjadi DataFrame bersih.

    Parameters
    ----------
    adm4_code : str
        Kode wilayah tingkat kelurahan (misal '32.76.06.1004' untuk Pondok Cina).
    timeout : int, default=10
        Batas waktu request HTTP dalam detik.

    Returns
    -------
    pd.DataFrame
        Kolom: ['timestamp', 'cuaca_desc', 'suhu', 'kelembapan', 'kecepatan_angin'].
        Mengembalikan DataFrame kosong berkolom sama jika API BMKG gagal/down.
    """
    df, _ = fetch_bmkg_with_status(adm4_code=adm4_code, timeout=timeout)
    return df


if __name__ == "__main__":
    sample_adm4 = DEPOK_LOCATIONS["Pondok Cina (UI Depok)"]["adm4"]
    df_bmkg, warning_msg = fetch_bmkg_with_status(sample_adm4)
    if warning_msg:
        print(f"[PERINGATAN]: {warning_msg}")
    print(f"Jumlah baris data BMKG: {len(df_bmkg)}")
    print(df_bmkg.head(8))

# -*- coding: utf-8 -*-
"""SMA data cache management (CSV zip format)."""

import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import SMA_CACHE_DIR


class SMACache:
    """Manage SMA data cache as CSV zip files."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else SMA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, dataframes: dict[str, pd.DataFrame], meta: dict = None) -> str:
        """
        Save SMA raw dataframes to a CSV zip file.
        Returns path to the created zip file.
        """
        meta = meta or {}
        meta_out = {
            "exported_at": datetime.now().isoformat(),
            **meta,
            "keys": sorted(list(dataframes.keys())),
        }

        date_str = datetime.now().strftime("%Y%m%d")
        zip_path = self.cache_dir / f"sma_cache_{date_str}.zip"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta_out, ensure_ascii=False, indent=2))

            for k, df in dataframes.items():
                if df is None or (hasattr(df, "empty") and df.empty):
                    continue
                tmp = df.copy()
                if "Date" in tmp.columns:
                    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                csv_bytes = tmp.to_csv(index=False).encode("utf-8-sig")
                zf.writestr(f"{k}.csv", csv_bytes)

        with open(zip_path, "wb") as f:
            f.write(buf.getvalue())

        return str(zip_path)

    def load(self, zip_path: str = None) -> tuple[dict[str, pd.DataFrame], dict]:
        """
        Load SMA data from a CSV zip file.
        If zip_path is None, loads the latest cache file.
        Returns (dataframes_dict, meta_dict).
        """
        if zip_path is None:
            zip_path = self.get_latest_path()
            if zip_path is None:
                return {}, {}

        sma = {}
        meta = {}
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            if "meta.json" in zf.namelist():
                meta = json.loads(zf.read("meta.json").decode("utf-8"))

            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                key = os.path.splitext(os.path.basename(name))[0]
                df = pd.read_csv(io.BytesIO(zf.read(name)))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                sma[key] = df

        return sma, meta

    def get_latest_path(self) -> str | None:
        """Return path to the most recent cache zip file, or None."""
        zips = sorted(self.cache_dir.glob("sma_cache_*.zip"), reverse=True)
        return str(zips[0]) if zips else None

    def is_fresh(self, max_age_days: int = 7) -> bool:
        """Check if the latest cache is within max_age_days."""
        latest = self.get_latest_path()
        if latest is None:
            return False
        # Parse date from filename
        try:
            fname = Path(latest).stem  # sma_cache_20260207
            date_str = fname.split("_")[-1]
            cache_date = datetime.strptime(date_str, "%Y%m%d").date()
            age = (datetime.now().date() - cache_date).days
            return age <= max_age_days
        except Exception:
            return False

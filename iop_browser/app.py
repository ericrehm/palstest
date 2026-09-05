"""
IOP Browser: a small read-only Flask app for browsing ac-s cast profiles
under IOPS/ -- both the raw acsPROdowncast-*.txt casts (no lat/lon) and
the acsPROdowncast-*.csv casts geolocate_iops.py produces (lat/lon added).

One route renders the page; two small JSON endpoints back it:
  /api/wavelengths -- the two wavelength axes, loaded once (static, doesn't
                       vary per cast)
  /api/files       -- every browsable cast under IOPS/
  /api/cast        -- one cast's full per-row data (sequence, pressure,
                       datetime, lat/lon if present, and both spectra) --
                       sent whole rather than paged, so the browser can do
                       cursor interpolation/redraws locally without a
                       server round trip on every drag frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

# So `iop_geo` (the project-root package) is importable when this is run
# directly, same pattern as v2/app_v2.py.
sys.path.insert(0, str(Path(__file__).parent.parent))
from iop_geo import load_wavelengths, load_acs_cast

IOPS_DIR = Path(__file__).parent.parent / "IOPS"

app = Flask(__name__)

A_WAVELENGTHS = load_wavelengths(IOPS_DIR / "acs-A-wavelengths-col.txt")
C_WAVELENGTHS = load_wavelengths(IOPS_DIR / "acs-C-wavelengths-col.txt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/wavelengths")
def wavelengths():
    return jsonify({"a": A_WAVELENGTHS, "c": C_WAVELENGTHS})


@app.route("/api/files")
def files():
    """Every browsable cast under IOPS/ -- the geolocated .csv (lat/lon
    included) and the raw .txt (lat/lon not available) alike."""
    entries = [{"name": p.name, "kind": "csv"} for p in IOPS_DIR.glob("acsPROdowncast-*.csv")]
    entries += [{"name": p.name, "kind": "txt"} for p in IOPS_DIR.glob("acsPROdowncast-*.txt")]
    entries.sort(key=lambda e: e["name"])
    return jsonify(entries)


def _resolve_cast_path(name: str) -> Path | None:
    """name must be a bare filename (no path separators) directly under
    IOPS_DIR matching the acsPROdowncast naming convention -- rejects path
    traversal and anything outside the expected cast files, even though
    this is a local-only read-only tool."""
    if not name or Path(name).name != name:
        return None
    if not name.startswith("acsPROdowncast-") or Path(name).suffix not in (".csv", ".txt"):
        return None
    path = IOPS_DIR / name
    return path if path.is_file() else None


@app.route("/api/cast")
def cast():
    path = _resolve_cast_path(request.args.get("name", ""))
    if path is None:
        return jsonify({"error": "Unknown cast file"}), 400

    try:
        if path.suffix == ".csv":
            # Already geolocated by geolocate_iops.py -- sequence, pressure,
            # datetime, latitude, longitude, c_*, a_* columns, in that order.
            df = pd.read_csv(path)
            has_lat_lon = "latitude" in df.columns
        else:
            df = load_acs_cast(path, C_WAVELENGTHS, A_WAVELENGTHS)
            has_lat_lon = False
    except Exception as e:
        return jsonify({"error": f"Failed to load {path.name}: {e}"}), 400

    if "latitude" not in df.columns:
        df["latitude"] = np.nan
        df["longitude"] = np.nan

    # datetime is already an ISO string on disk for the .csv path (written
    # by geolocate_iops.py) but a tz-aware Timestamp column for the .txt
    # path (load_acs_cast's own return type) -- normalize both to the same
    # ISO string shape the client expects.
    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        dt_strs = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    else:
        dt_strs = df["datetime"].astype(str)

    a_cols = [c for c in df.columns if c.startswith("a_")]
    c_cols = [c for c in df.columns if c.startswith("c_")]
    a_matrix = df[a_cols].to_numpy()
    c_matrix = df[c_cols].to_numpy()
    sequence = df["sequence"].to_numpy()
    pressure = df["pressure"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    def _row_list(arr):
        return [None if np.isnan(v) else float(v) for v in arr]

    rows = [
        {
            "sequence": int(sequence[i]),
            "pressure": float(pressure[i]),
            "datetime": dt_strs.iloc[i],
            "lat": None if np.isnan(lat[i]) else float(lat[i]),
            "lon": None if np.isnan(lon[i]) else float(lon[i]),
            "a": _row_list(a_matrix[i]),
            "c": _row_list(c_matrix[i]),
        }
        for i in range(len(df))
    ]

    return jsonify({"filename": path.name, "has_lat_lon": has_lat_lon, "rows": rows})


if __name__ == "__main__":
    # 5025 (v2) and 5050 (v1) are already claimed; 5000 is macOS AirPlay
    # Receiver (see v2/app_v2.py's note) -- 5040 avoids all three.
    app.run(debug=True, port=5040)

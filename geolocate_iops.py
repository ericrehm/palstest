"""
Interpolate ship lat/lon (outgoing_data/*.csv, 1-min samples) onto each
ac-s IOP downcast's per-row UTC datetime (IOPS/acsPROdowncast-*.txt), and
write each cast out as a .csv alongside the original .txt.

Usage:
    python3 geolocate_iops.py [--iops-dir IOPS] [--nav-dir outgoing_data]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iop_geo import (
    load_wavelengths,
    load_acs_cast,
    load_nav_directory,
    clean_lat_lon,
    DATETIME_COL,
    LAT_COL,
    LON_COL,
)

# True ISO-8601 (the 'T' separator) -- pandas' own to_csv default for a
# tz-aware datetime64 column uses a space instead ("2026-08-30
# 09:06:49.008000+0000"), which is a valid ISO 8601 alternate form but not
# what "ISO format" usually means to a reader.
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"


def interpolate_lat_lon(cast_datetimes: pd.Series, nav: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation of nav's Latitude/Longitude onto cast_datetimes.

    np.interp clamps to the nearest endpoint value for a timestamp outside
    nav's own range rather than extrapolating or raising -- fine as long as
    the cast is actually within the cruise's nav coverage (checked by the
    caller), since a clamped value silently reads as a real position
    otherwise.
    """
    nav_ns = nav[DATETIME_COL].astype("int64").to_numpy()
    cast_ns = cast_datetimes.astype("int64").to_numpy()
    lat = np.interp(cast_ns, nav_ns, nav[LAT_COL].to_numpy())
    lon = np.interp(cast_ns, nav_ns, nav[LON_COL].to_numpy())
    return pd.DataFrame({"latitude": lat, "longitude": lon})


def geolocate_cast(cast_path: Path, nav: pd.DataFrame, c_wl: list[float], a_wl: list[float]) -> pd.DataFrame:
    cast = load_acs_cast(cast_path, c_wl, a_wl)

    out_of_range = (cast["datetime"] < nav[DATETIME_COL].iloc[0]) | (cast["datetime"] > nav[DATETIME_COL].iloc[-1])
    if out_of_range.any():
        print(
            f"  WARNING: {out_of_range.sum()}/{len(cast)} rows fall outside the nav "
            f"data's time range ({nav[DATETIME_COL].iloc[0]} to {nav[DATETIME_COL].iloc[-1]}) "
            f"-- their lat/lon is clamped to the nearest endpoint, not a real interpolation."
        )

    lat_lon = interpolate_lat_lon(cast["datetime"], nav)

    value_cols = [c for c in cast.columns if c not in ("sequence", "pressure", "datetime")]
    return pd.concat(
        [cast[["sequence", "pressure", "datetime"]], lat_lon, cast[value_cols]], axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iops-dir", type=Path, default=Path("IOPS"), help="Directory of acsPROdowncast-*.txt casts")
    parser.add_argument("--nav-dir", type=Path, default=Path("outgoing_data"), help="Directory of ship nav *.csv files")
    parser.add_argument(
        "--a-wavelengths", type=Path, default=None,
        help="Defaults to <iops-dir>/acs-A-wavelengths-col.txt",
    )
    parser.add_argument(
        "--c-wavelengths", type=Path, default=None,
        help="Defaults to <iops-dir>/acs-C-wavelengths-col.txt",
    )
    args = parser.parse_args()

    a_wl_path = args.a_wavelengths or (args.iops_dir / "acs-A-wavelengths-col.txt")
    c_wl_path = args.c_wavelengths or (args.iops_dir / "acs-C-wavelengths-col.txt")
    a_wl = load_wavelengths(a_wl_path)
    c_wl = load_wavelengths(c_wl_path)
    print(f"Loaded {len(a_wl)} a(lambda) and {len(c_wl)} c(lambda) wavelengths")

    print(f"Loading nav data from {args.nav_dir} ...")
    nav = clean_lat_lon(load_nav_directory(args.nav_dir))
    print(f"  {len(nav)} nav samples, {nav[DATETIME_COL].iloc[0]} to {nav[DATETIME_COL].iloc[-1]}")

    cast_paths = sorted(args.iops_dir.glob("acsPROdowncast-*.txt"))
    if not cast_paths:
        raise FileNotFoundError(f"No acsPROdowncast-*.txt files found in {args.iops_dir}")

    print(f"Geolocating {len(cast_paths)} cast file(s)...")
    for cast_path in cast_paths:
        print(f"  {cast_path.name}")
        out = geolocate_cast(cast_path, nav, c_wl, a_wl)
        out_path = cast_path.with_suffix(".csv")
        out.to_csv(out_path, index=False, date_format=_ISO_FMT)
        print(f"    -> {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()

"""
Ship navigation data (outgoing_data/*.csv), 1-minute samples.

Each file is one hour's worth of rows (61 columns total; only Date/Time,
Latitude, and Longitude matter here, but every column is kept -- see
load_nav_file). Loading the whole directory concatenates every file into
one time-sorted dataframe spanning the full cruise, since an IOP cast can
fall on any date/time in it and interpolation (see geolocate_iops.py)
needs the whole timeline, not just one file's hour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATETIME_COL = "Date/Time"
LAT_COL = "Latitude (degree_north)"
LON_COL = "Longitude (degree_east)"

# Same sentinel convention as the ac-s files (see acs.MISSING_VALUE), seen
# on Latitude/Longitude on rare bad-GPS-lock rows -- must be dropped before
# interpolating on lat/lon, not just left as a garbage value in range.
_MISSING_VALUE = -999.9


def load_nav_file(path: str | Path) -> pd.DataFrame:
    """One outgoing_data/*.csv file, every column, with Date/Time parsed
    to a tz-aware UTC timestamp."""
    df = pd.read_csv(path)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], utc=True)
    return df


def load_nav_directory(directory: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Every file matching pattern in directory, concatenated and sorted
    by Date/Time into one dataframe."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    nav = pd.concat((load_nav_file(f) for f in files), ignore_index=True)
    nav = nav.sort_values(DATETIME_COL).reset_index(drop=True)
    return nav


def clean_lat_lon(nav: pd.DataFrame) -> pd.DataFrame:
    """nav with any row dropped whose Latitude/Longitude is the -999.9
    missing-value sentinel -- interpolation (np.interp) has no concept of
    NaN passthrough, so a bad sample left in would quietly poison every
    IOP timestamp that lands near it."""
    bad = np.isclose(nav[LAT_COL], _MISSING_VALUE) | np.isclose(nav[LON_COL], _MISSING_VALUE)
    return nav.loc[~bad].reset_index(drop=True)

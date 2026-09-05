"""
ac-s IOP downcast profiles (IOPS/acsPROdowncast-YYYYMMDD-castN.txt).

Whitespace-delimited, no header row. Each row is:
    Sequence, Pressure, Time (hh:mm:ss.fff), <c(lambda_i)> x M, <a(lambda_j)> x N

M and N are the row counts of acs-C-wavelengths-col.txt and
acs-A-wavelengths-col.txt respectively (one wavelength, in nm, per line,
in column order). -999 marks a missing value.

The file's own Time column has no date -- the cast's UTC date is only
available in the filename (the YYYYMMDD run in
"acsPROdowncast-20260830-cast4.txt") -- so load_acs_cast merges the two
into one UTC datetime column, replacing the raw Time column in place (same
position: 3rd column).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd

MISSING_VALUE = -999.0

_CAST_DATE_RE = re.compile(r"(\d{8})")


def load_wavelengths(path: str | Path) -> list[float]:
    """One float wavelength (nm) per line, in column order."""
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def _cast_date_from_filename(path: Path) -> pd.Timestamp:
    """The cast's UTC date, from the 8-digit YYYYMMDD run in the filename.
    Raises if the filename doesn't carry one -- there's no other source
    for the date, so silently defaulting would misdate every row."""
    m = _CAST_DATE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Could not find an 8-digit YYYYMMDD date in filename: {path.name}")
    return pd.Timestamp(m.group(1), tz="UTC")


def load_acs_cast(
    path: str | Path,
    c_wavelengths: Sequence[float],
    a_wavelengths: Sequence[float],
) -> pd.DataFrame:
    """
    Load one acsPROdowncast .txt file into a DataFrame.

    Columns, in order: 'sequence', 'pressure', 'datetime' (tz-aware UTC,
    merged from the filename's date + the row's time-of-day -- see module
    docstring), then one 'c_<wavelength>' column per c_wavelengths entry,
    then one 'a_<wavelength>' column per a_wavelengths entry. -999 becomes
    NaN, but only within the c_/a_ columns -- sequence/pressure/time are
    never sentinel-valued in these files, and blindly matching -999
    against every column risks nuking a legitimate one.
    """
    path = Path(path)
    c_cols = [f"c_{wl:.1f}" for wl in c_wavelengths]
    a_cols = [f"a_{wl:.1f}" for wl in a_wavelengths]
    value_cols = c_cols + a_cols
    columns = ["sequence", "pressure", "time"] + value_cols

    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)

    expected_cols = 3 + len(c_wavelengths) + len(a_wavelengths)
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"{path.name}: expected {expected_cols} columns "
            f"(3 + {len(c_wavelengths)} c + {len(a_wavelengths)} a), got {df.shape[1]}"
        )

    df[value_cols] = df[value_cols].mask(df[value_cols] == MISSING_VALUE)

    cast_date = _cast_date_from_filename(path)
    datetime_col = cast_date + pd.to_timedelta(df["time"])
    datetime_col.name = "datetime"

    return pd.concat([df[["sequence", "pressure"]], datetime_col, df[value_cols]], axis=1)

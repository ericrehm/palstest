"""
Matching ac-s IOP casts to PALS lidar shots by time, and depth-averaging
c(lambda) over a range window -- the link between L2's K_lidar fit windows
(v2/config/PALS_SBS312.json's k_lidar_fit_window_*_m) and the in-situ ac-s
casts under IOPS/.

Two pieces:
  IOPProfileSet   -- every acsPROdowncast-*.txt cast, loaded once, each
                      row's c(lambda) spectrum pre-interpolated to exactly
                      532nm and 650nm (wavelength-only, independent of any
                      fit window -- safe to load once and reuse). Answers
                      "which cast (if any) is contemporaneous with this
                      lidar shot's timestamp?".
  IOPWindowAverage -- one IOPProfileSet + a caller-supplied set of depth
                      windows and c_water baseline (typically an
                      L2Processor's own l2_constants -- same windows that
                      produced K_lidar/c_est, including any per-request
                      override, so the IOP columns compare against exactly
                      what was fit). Answers "for this lidar shot's
                      timestamp, what's the depth-averaged, c_water-
                      corrected total attenuation ct532/ct650 over each
                      window, in its contemporaneous cast?" -- memoized per
                      cast, since the same cast backs many consecutive
                      shots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .acs import load_acs_cast, load_wavelengths

# A lidar shot counts as contemporaneous with a cast if its timestamp
# falls within the cast's own [first, last] row time, padded by this much
# on each end -- not a symmetric "+/- 20 min from the profile", which
# would double-count the cast's own (usually short, a few minutes)
# duration.
MATCH_PAD = pd.Timedelta(minutes=10)

# output column name -> (target wavelength nm, L2Constants field suffix).
# The suffix is used as "k_lidar_fit_window_{suffix}_m" to look up the
# matching window off whatever L2Processor produced the K_lidar fit these
# IOP values are meant to sit alongside. "ct" (not "c") because the ac-s's
# own c(lambda) is particulate/CDOM only (factory-calibrated against pure
# water) -- IOPWindowAverage adds back c_water_{532,650} so these are total
# beam attenuation, directly comparable to K_lidar (also a total-
# attenuation estimate), not to c_est (which is K_lidar with c_water
# already subtracted back out).
OUTPUT_SPECS: Tuple[Tuple[str, float, str], ...] = (
    ("ct532_near", 532.0, "nf"),
    ("ct532_far", 532.0, "ff"),
    ("ct650_near", 650.0, "raman_nf"),
    ("ct650_far", 650.0, "raman_ff"),
)


def interp_rows_to_wavelength(wavelengths: np.ndarray, values: np.ndarray, target: float) -> np.ndarray:
    """Linear-interpolate each row of `values` (shape (n_rows, n_wavelengths))
    at a single fixed `target` wavelength, given the shared, sorted-ascending
    `wavelengths` (n_wavelengths,) axis every row shares.

    Vectorized across rows rather than one np.interp call per row: the
    bracketing wavelength pair is the same for every row (the wavelength
    axis doesn't vary row to row), so it's found once via searchsorted and
    then the actual interpolation is one array-wide computation. NaNs in
    `values` propagate through the arithmetic untouched.
    """
    idx = np.searchsorted(wavelengths, target)
    idx = np.clip(idx, 1, len(wavelengths) - 1)
    x0, x1 = wavelengths[idx - 1], wavelengths[idx]
    y0, y1 = values[:, idx - 1], values[:, idx]
    frac = (target - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


@dataclass
class _Profile:
    name: str
    datetime: np.ndarray  # datetime64[ns, UTC], ascending
    pressure: np.ndarray
    c_at: Dict[float, np.ndarray]  # target wavelength (e.g. 532.0) -> per-row interpolated c(lambda)
    first_time: pd.Timestamp
    last_time: pd.Timestamp

    @property
    def center_time(self) -> pd.Timestamp:
        return self.first_time + (self.last_time - self.first_time) / 2


class IOPProfileSet:
    """Every acsPROdowncast-*.txt cast under `iops_dir`, loaded once.

    Each cast's c(lambda) spectrum is pre-interpolated per row to every
    wavelength named in OUTPUT_SPECS (currently 532/650nm) at load time --
    that interpolation only depends on the cast's own data, never on a fit
    window, so it's safe to do once up front rather than repeating it on
    every match.
    """

    def __init__(self, iops_dir: str | Path):
        iops_dir = Path(iops_dir)
        a_wl = load_wavelengths(iops_dir / "acs-A-wavelengths-col.txt")
        c_wl = load_wavelengths(iops_dir / "acs-C-wavelengths-col.txt")
        c_wl_arr = np.asarray(c_wl)
        targets = sorted({wl for _, wl, _ in OUTPUT_SPECS})

        self.profiles: list[_Profile] = []
        for cast_path in sorted(iops_dir.glob("acsPROdowncast-*.txt")):
            cast = load_acs_cast(cast_path, c_wl, a_wl)
            c_matrix = cast[[c for c in cast.columns if c.startswith("c_")]].to_numpy()
            c_at = {wl: interp_rows_to_wavelength(c_wl_arr, c_matrix, wl) for wl in targets}
            dt = cast["datetime"].to_numpy()
            self.profiles.append(_Profile(
                name=cast_path.stem,
                datetime=dt,
                pressure=cast["pressure"].to_numpy(),
                c_at=c_at,
                first_time=pd.Timestamp(dt.min()),
                last_time=pd.Timestamp(dt.max()),
            ))

    def find_contemporaneous(self, timestamp_utc: datetime) -> Optional[_Profile]:
        """The cast whose own [first, last] row time, padded by MATCH_PAD on
        each end, contains timestamp_utc -- or None. If more than one cast
        qualifies (casts scheduled close enough together that their padded
        windows overlap), the one whose center_time is nearest wins."""
        candidates = [
            p for p in self.profiles
            if (p.first_time - MATCH_PAD) <= timestamp_utc <= (p.last_time + MATCH_PAD)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return min(candidates, key=lambda p: abs(p.center_time - timestamp_utc))


class IOPWindowAverage:
    """IOPProfileSet + a set of depth windows -> per-shot ct532/ct650 near/far.

    windows: {"nf": (r1, r2), "ff": (r1, r2), "raman_nf": (r1, r2),
    "raman_ff": (r1, r2)} -- pass an L2Processor's own
    l2_constants.k_lidar_fit_window_{suffix}_m values so the IOP averages
    line up with whatever windows actually produced that request's
    K_lidar/c_est (including any per-request UI override).

    c_water: {532.0: c_water_532, 650.0: c_water_650} (an L2Processor's own
    l2_constants) -- added to each window's depth-averaged ac-s c(lambda)
    to turn it from particulate/CDOM-only into total beam attenuation (see
    OUTPUT_SPECS's docstring note on "ct" vs "c").

    Per-cast depth averages are memoized (keyed by cast name) for the
    lifetime of this instance -- build one per L2 request/response, not
    once globally, since a global instance would go stale the moment a
    request overrides the fit windows or c_water.
    """

    def __init__(
        self,
        profile_set: IOPProfileSet,
        windows: Dict[str, Tuple[float, float]],
        c_water: Dict[float, float],
    ):
        self._profiles = profile_set
        self._windows = windows
        self._c_water = c_water
        self._cache: Dict[str, Dict[str, Optional[float]]] = {}

    def _averages_for_profile(self, profile: _Profile) -> Dict[str, Optional[float]]:
        cached = self._cache.get(profile.name)
        if cached is not None:
            return cached

        result: Dict[str, Optional[float]] = {}
        for out_name, wavelength, window_key in OUTPUT_SPECS:
            window = self._windows.get(window_key)
            if window is None:
                result[out_name] = None
                continue
            r1, r2 = window
            in_window = (profile.pressure >= r1) & (profile.pressure <= r2)
            if not np.any(in_window):
                result[out_name] = None
                continue
            values = profile.c_at[wavelength][in_window]
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(values)
            result[out_name] = None if np.isnan(mean) else float(mean + self._c_water.get(wavelength, 0.0))

        self._cache[profile.name] = result
        return result

    def values_for(self, timestamp_utc: datetime) -> Dict[str, Optional[float]]:
        """{"ct532_near", "ct532_far", "ct650_near", "ct650_far"} -> value or
        None (no contemporaneous cast, or that window had no in-range/
        valid rows in the matched cast)."""
        profile = self._profiles.find_contemporaneous(timestamp_utc)
        if profile is None:
            return {out_name: None for out_name, _, _ in OUTPUT_SPECS}
        return self._averages_for_profile(profile)

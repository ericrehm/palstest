"""
In-memory data store for the quick-and-dirty Plotly viewer (raw/L0b/L1/L2).

Deliberately a single "currently loaded file" slot, not a multi-user session
store -- this is a local inspection tool, not the processing pipeline. The
whole file is parsed once on load (so windowed requests are cheap numpy
slices), but a window request never returns more than the caller's requested
time slice -- shipping a whole L1/L2 cruise file's matrix as JSON is exactly
what blew up /api/process-file's old combined-JSON response (see app_v2.py).
"""
import bisect
import csv
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .io import parse_l1_shots
from .io.pals_io import KNOWN_IDENTITIES, parse_pmt_gain_header


def _sanitize(arr) -> list:
    """
    Replace inf/-inf/nan with None so json.dumps produces valid JSON.
    Ratio identities (qer_*/depol_*) are inf or nan wherever a shot's
    denominator channel is zero -- by construction, not a bug -- but
    Python's default JSON encoder emits those as bare Infinity/NaN tokens,
    which JS's JSON.parse rejects outright ("not valid JSON"). Works for
    both 1D and 2D arrays.
    """
    arr = np.asarray(arr, dtype=float)
    obj = arr.astype(object)
    obj[~np.isfinite(arr)] = None
    return obj.tolist()


@dataclass
class ViewerIdentity:
    timestamps: np.ndarray  # int64 epoch ms, shape (n_shots,), sorted ascending
    z: np.ndarray  # float64, shape (n_shots, n_bins)
    axis: np.ndarray  # float64, shape (n_bins,) -- bin index (raw/L0b) or range_m (L1/L2)
    axis_label: str  # "Bin" or "Range (m)"


@dataclass
class ViewerFile:
    file_type: str  # 'raw' | 'l0b' | 'l1' | 'l2'
    filename: str
    identities: Dict[str, ViewerIdentity] = field(default_factory=dict)
    # {gain index (1-6): setpoint volts}, parsed from the file's own
    # '# pmt_gain_N: <value>' header lines -- an operational per-run
    # recording, not calibration data, so it lives in the file, not the
    # instrument config. Empty if the header had none (shouldn't normally
    # happen for real files, but downstream code treats it as "unknown").
    pmt_gain_by_index: Dict[int, float] = field(default_factory=dict)

    def meta(self) -> dict:
        starts = [ident.timestamps[0] for ident in self.identities.values() if len(ident.timestamps)]
        ends = [ident.timestamps[-1] for ident in self.identities.values() if len(ident.timestamps)]
        return {
            "file_type": self.file_type,
            "filename": self.filename,
            "identities": sorted(self.identities.keys()),
            "start_ms": int(min(starts)) if starts else 0,
            "end_ms": int(max(ends)) if ends else 0,
            "axis_labels": {name: ident.axis_label for name, ident in self.identities.items()},
            "axes": {name: ident.axis.tolist() for name, ident in self.identities.items()},
            "shot_counts": {name: int(len(ident.timestamps)) for name, ident in self.identities.items()},
            # String keys -- JSON object keys can't be integers.
            "pmt_gain_by_index": {str(k): v for k, v in self.pmt_gain_by_index.items()},
            # The six real PMT channels, as opposed to L1CrossProcessor's
            # ratio identities, L2's ensemble-SNR rows, or the QA
            # pseudo-identities ("<channel>:<test>") -- PMT gain only means
            # anything for these.
            "natural_identities": sorted(KNOWN_IDENTITIES),
        }

    def window(self, identity: str, start_ms: int, end_ms: int) -> dict:
        ident = self.identities.get(identity)
        if ident is None:
            return {"timestamps": [], "z": []}
        lo = bisect.bisect_left(ident.timestamps, start_ms)
        hi = bisect.bisect_right(ident.timestamps, end_ms)
        z_slice = ident.z[lo:hi]  # shape (count, n_bins)
        return {
            # Transposed to (n_bins, count) -- Plotly heatmap's z[row][col]
            # convention, where row = y-axis (bin/range), col = x-axis (time).
            "timestamps": ident.timestamps[lo:hi].tolist(),
            "z": _sanitize(z_slice.T),
        }

    def nearest(self, identity: str, target_ms: int) -> dict:
        """
        Single shot's waveform nearest to target_ms, for one identity. Used
        by the compare page's per-file time sliders -- returns exactly one
        waveform, not a window of them, since only one column is ever needed
        at a time there.
        """
        ident = self.identities.get(identity)
        if ident is None or not len(ident.timestamps):
            return {"timestamp_ms": None, "z": []}
        idx = int(np.argmin(np.abs(ident.timestamps - target_ms)))
        return {"timestamp_ms": int(ident.timestamps[idx]), "z": _sanitize(ident.z[idx])}

    def average(self, identity: str) -> dict:
        """
        Ensemble average waveform for one identity across every shot in the
        whole file. The full z-matrix is already resident in memory from
        load() -- computing the mean over it is effectively free -- so this
        returns just one waveform (n_bins floats), not the matrix used to
        compute it.

        Uses nanmean, not a plain mean: ratio identities (qer_*/depol_*) can
        be inf/nan at a bin for individual shots (denominator was zero for
        that shot only) -- a plain mean would let one such shot poison the
        whole bin's average to nan, same reasoning as L2Processor.process().
        """
        ident = self.identities.get(identity)
        if ident is None or not len(ident.z):
            return {"z": []}
        z = np.where(np.isfinite(ident.z), ident.z, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg = np.nanmean(z, axis=0)
        return {"z": _sanitize(avg)}


_METADATA_PREFIX = "#"


def load_raw_or_l0b(lines: List[str], filename: str) -> ViewerFile:
    """
    Parse the real (multi-Identity-per-row) raw/L0b "Matrix" CSV format.
    Adapted from pypals.py's PalsData._load_one_stream -- raw and L0b share
    the exact same on-disk shape (L0b just adds matched PowerASum/PowerBSum
    columns), so one parser handles both; only Timestamp/Identity/Bin_ columns
    matter for viewing.
    """
    lines_iter = iter(lines)
    header = None
    metadata_lines: List[str] = []
    for raw_line in lines_iter:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_METADATA_PREFIX):
            metadata_lines.append(line)
            continue
        header = [c.strip() for c in line.split(",")]
        break
    if not header:
        raise ValueError(f"No CSV header found in {filename}")

    bin_cols = [c for c in header if c.startswith("Bin_")]
    if not bin_cols:
        raise ValueError(f"No Bin_ columns found in {filename}")

    by_identity: Dict[str, List[tuple]] = {}
    reader = csv.DictReader(lines_iter, fieldnames=header)
    for record in reader:
        identity = (record.get("Identity") or "unknown").strip()
        try:
            ts = int(float(record.get("Timestamp") or 0))
        except (TypeError, ValueError):
            continue
        try:
            waveform = [float(record.get(c) or 0) for c in bin_cols]
        except (TypeError, ValueError):
            continue
        by_identity.setdefault(identity, []).append((ts, waveform))

    axis = np.arange(len(bin_cols), dtype=float)
    identities: Dict[str, ViewerIdentity] = {}
    for identity, rows in by_identity.items():
        rows.sort(key=lambda r: r[0])
        timestamps = np.array([r[0] for r in rows], dtype=np.int64)
        z = np.array([r[1] for r in rows], dtype=float)
        identities[identity] = ViewerIdentity(timestamps=timestamps, z=z, axis=axis, axis_label="Bin")

    pmt_gain_by_index = parse_pmt_gain_header(metadata_lines)
    return ViewerFile(file_type="raw", filename=filename, identities=identities, pmt_gain_by_index=pmt_gain_by_index)


def _epoch_ms(dt) -> int:
    return int(dt.timestamp() * 1000)


def load_l1_or_l2(lines: List[str], filename: str) -> ViewerFile:
    """
    L1 and L2 share the same on-disk column layout (see io.parse_l1_shots),
    so both go through this reader. Exposes each channel's per-bin QA value
    arrays (e.g. detector SNR, eqn 9's Q_ADC) as their own selectable
    pseudo-identities named "<channel>:<test>" -- reuses the same viewer
    machinery to look at SNR/QA alongside the raw signal, no special-casing
    needed. L2's ensemble-SNR rows (synthetic identities like
    "co_near_ensemble_snr") come through parse_l1_shots as ordinary
    identities already, so they "just work" here too.
    """
    shots = parse_l1_shots(lines, source_file=filename)
    if not shots:
        raise ValueError(f"No recognized shots found in {filename}")

    by_identity: Dict[str, List[tuple]] = {}
    for shot in shots:
        channel_id = next(iter(shot.signal))
        ts = _epoch_ms(shot.l0b.l0.timestamp_utc)
        range_m = shot.range_m.get(channel_id, np.array([]))
        by_identity.setdefault(channel_id, []).append((ts, shot.signal[channel_id], range_m))
        for test_name, per_channel in shot.qa_values.items():
            if channel_id in per_channel:
                key = f"{channel_id}:{test_name}"
                by_identity.setdefault(key, []).append((ts, per_channel[channel_id], range_m))

    identities: Dict[str, ViewerIdentity] = {}
    for identity, rows in by_identity.items():
        rows.sort(key=lambda r: r[0])
        timestamps = np.array([r[0] for r in rows], dtype=np.int64)
        z = np.array([r[1] for r in rows], dtype=float)
        axis = np.asarray(rows[0][2], dtype=float)
        identities[identity] = ViewerIdentity(timestamps=timestamps, z=z, axis=axis, axis_label="Range (m)")

    # header_lines (including the original raw file's '# pmt_gain_N:' lines)
    # are carried forward into L1/L2 files by write_l1_shots and read back by
    # parse_l1_shots -- same header, same parser, regardless of stage.
    header_lines = shots[0].l0b.l0.metadata.get('header_lines', [])
    pmt_gain_by_index = parse_pmt_gain_header(header_lines)

    # file_type is overwritten by the caller (it knows which stage the user
    # said they were loading); 'l1' here is just a harmless default.
    return ViewerFile(
        file_type="l1", filename=filename, identities=identities, pmt_gain_by_index=pmt_gain_by_index
    )

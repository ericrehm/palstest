from __future__ import annotations

import bisect
import csv
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


METADATA_PREFIX = "#"
PULSE_OFFSET_MS = 50
GROUP_NEAR_PAIR = "near_pair"
GROUP_FAR_PAIR = "far_pair"
GROUP_RAMAN_PAIR = "raman_pair"


def default_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "testdata" / "Test1_Hotel_12.csv"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    # Return None if either parameter is None
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _tilt_from_roll_pitch(roll_deg: float, pitch_deg: float) -> float:
    # Tilt is computed from roll/pitch as a single inclination angle from vertical.
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    cos_tilt = math.cos(roll_rad) * math.cos(pitch_rad)
    cos_tilt = max(-1.0, min(1.0, cos_tilt))
    return math.degrees(math.acos(cos_tilt))


def apply_gaussian_filter_2d(z_matrix: list[list[float]], sigma_v: float, sigma_h: float) -> list[list[float]]:
    """
    Apply 2D Gaussian filtering to a heatmap matrix.
    
    Args:
        z_matrix: 2D list where each row is a bin and each column is a timestamp
        sigma_v: Gaussian sigma in vertical (bin/range) dimension
        sigma_h: Gaussian sigma in horizontal (pulse time) dimension
    
    Returns:
        Filtered 2D list with same shape as input
    """
    # If filter is effectively disabled (very small sigmas), return original
    if sigma_v < 0.01 and sigma_h < 0.01:
        return z_matrix
    
    # Convert to numpy array
    data = np.array(z_matrix, dtype=np.float64)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    data_safe = np.maximum(data, epsilon)
    
    # Log transform
    log_data = np.log(data_safe)
    
    # Apply Gaussian filter
    filtered_log_data = ndimage.gaussian_filter(log_data, sigma=[sigma_v, sigma_h], mode='nearest')
    
    # Undo log transform
    filtered_data = np.exp(filtered_log_data)
    
    # Convert back to list of lists and preserve NaN/None values
    result = filtered_data.tolist()
    return result


class PalsData:
    def __init__(self, csv_path: str | Path | list[str | Path]):
        if isinstance(csv_path, list):
            self.csv_paths = [Path(item) for item in csv_path]
        else:
            self.csv_paths = [Path(csv_path)]

        self.csv_path = self.csv_paths[0]
        self.loaded_files: list[str] = []
        self.metadata_by_file: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.header: list[str] = []
        self.waveform_columns: list[str] = []
        self.samples_per_pulse: int = 0
        self.identities: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self._by_identity: dict[str, list[dict[str, Any]]] = {}
        self._timestamps_by_identity: dict[str, list[int]] = {}
        self.power_format: str = "unknown"  # "old" or "new"
        self._load()

    @classmethod
    def from_uploaded_files(cls, uploaded_files: list[Any]) -> "PalsData":
        instance = cls.__new__(cls)
        instance.csv_paths = []
        instance.csv_path = Path("uploaded")
        instance.loaded_files = []
        instance.metadata_by_file = {}
        instance.metadata = {}
        instance.header = []
        instance.waveform_columns = []
        instance.samples_per_pulse = 0
        instance.identities = []
        instance.rows = []
        instance._by_identity = {}
        instance._timestamps_by_identity = {}
        instance.power_format = "unknown"

        for uploaded in uploaded_files:
            name = (getattr(uploaded, "filename", None) or "uploaded.csv").strip() or "uploaded.csv"
            raw = uploaded.read()
            text = raw.decode("utf-8-sig", errors="replace")
            instance._load_one_stream(io.StringIO(text), name)

        instance._finalize()
        return instance

    def _load(self) -> None:
        for csv_path in self.csv_paths:
            if not csv_path.exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")

            with csv_path.open("r", newline="") as handle:
                self._load_one_stream(handle, csv_path.name)

        self._finalize()

    def _load_one_stream(self, handle: Any, source_name: str) -> None:
        file_metadata: dict[str, Any] = {}
        header_line = None
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(METADATA_PREFIX):
                key_value = line[1:].strip()
                if ":" in key_value:
                    key, value = key_value.split(":", 1)
                    file_metadata[key.strip()] = _parse_scalar(value)
                continue
            header_line = line
            break

        if not header_line:
            raise ValueError(f"CSV header not found after metadata section in {source_name}")

        header = [column.strip() for column in header_line.split(",")]
        waveform_columns = [c for c in header if c.startswith("Bin_")]
        if not waveform_columns:
            raise ValueError(f"No waveform columns found in {source_name}")

        # Detect power format based on header columns
        has_old_format = all(col in header for col in ["PowerAMin", "PowerAMax", "PowerBMin", "PowerBMax"])
        has_new_format = all(col in header for col in ["PowerASum", "PowerBSum"])
        
        if has_old_format:
            detected_format = "old"
        elif has_new_format:
            detected_format = "new"
        else:
            detected_format = "unknown"
        
        # Ensure all files use the same format
        if self.power_format == "unknown":
            self.power_format = detected_format
        elif self.power_format != detected_format:
            raise ValueError(
                f"Power format mismatch: expected {self.power_format}, got {detected_format} in {source_name}"
            )

        if not self.header:
            self.header = header
            self.waveform_columns = waveform_columns
            self.samples_per_pulse = len(self.waveform_columns)
        elif len(waveform_columns) != self.samples_per_pulse:
            raise ValueError(
                f"Waveform width mismatch in {source_name}: expected {self.samples_per_pulse}, got {len(waveform_columns)}"
            )

        reader = csv.DictReader(handle, fieldnames=header)
        for record in reader:
            timestamp_ms = _safe_int(record.get("Timestamp"))
            identity = (record.get("Identity") or "unknown").strip()
            roll = _safe_float(record.get("Roll"))
            pitch = _safe_float(record.get("Pitch"))
            yaw = _safe_float(record.get("Yaw"))

            waveform = [_safe_int(record.get(col)) for col in waveform_columns]
            if not waveform:
                continue

            # Parse power data based on detected format
            if detected_format == "old":
                row = {
                    "timestamp": timestamp_ms,
                    "identity": identity,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "tilt_deg": _tilt_from_roll_pitch(roll, pitch),
                    "laser_temp_c": _safe_float(record.get("LaserTempC")),
                    "power_a_min": _safe_float(record.get("PowerAMin")),
                    "power_a_max": _safe_float(record.get("PowerAMax")),
                    "power_b_min": _safe_float(record.get("PowerBMin")),
                    "power_b_max": _safe_float(record.get("PowerBMax")),
                    "waveform": waveform,
                }
            else:  # new format or unknown
                row = {
                    "timestamp": timestamp_ms,
                    "identity": identity,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "tilt_deg": _tilt_from_roll_pitch(roll, pitch),
                    "laser_temp_c": _safe_float(record.get("LaserTempC")),
                    "power_a_sum": _safe_float(record.get("PowerASum")),
                    "power_b_sum": _safe_float(record.get("PowerBSum")),
                    "waveform": waveform,
                }

            self.rows.append(row)
            self._by_identity.setdefault(identity, []).append(row)

        self.loaded_files.append(source_name)
        self.metadata_by_file[source_name] = file_metadata
        if not self.metadata:
            self.metadata = file_metadata

    def _finalize(self) -> None:
        self.identities = sorted(self._by_identity.keys())

        for identity in self.identities:
            rows = self._by_identity[identity]
            rows.sort(key=lambda item: item["timestamp"])
            self._timestamps_by_identity[identity] = [item["timestamp"] for item in rows]

    def get_metadata_payload(self) -> dict[str, Any]:
        file_label = self.loaded_files[0] if len(self.loaded_files) == 1 else f"{len(self.loaded_files)} files"
        return {
            "file": file_label,
            "files": self.loaded_files,
            "file_count": len(self.loaded_files),
            "metadata": self.metadata,
            "samples_per_pulse": self.samples_per_pulse,
            "total_rows": len(self.rows),
            "identities": self.identities,
        }

    def get_heatmaps_payload(self, sigma_v: float = 1.5, sigma_h: float = 4.0) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "samples_per_pulse": self.samples_per_pulse,
            "identities": self.identities,
            "series": {},
            "series_filtered": {},
        }

        for identity in self.identities:
            rows = self._by_identity.get(identity, [])
            timestamps = [r["timestamp"] for r in rows]
            waveforms = [r["waveform"] for r in rows]

            # Plotly heatmap expects a matrix where each row maps to one y-axis value.
            z_matrix = [
                [waveform[bin_index] for waveform in waveforms]
                for bin_index in range(self.samples_per_pulse)
            ]

            payload["series"][identity] = {
                "timestamps": timestamps,
                "bins": list(range(self.samples_per_pulse)),
                "z": z_matrix,
            }
            
            # Apply Gaussian filtering for filtered version
            z_matrix_filtered = apply_gaussian_filter_2d(z_matrix, sigma_v, sigma_h)
            payload["series_filtered"][identity] = {
                "timestamps": timestamps,
                "bins": list(range(self.samples_per_pulse)),
                "z": z_matrix_filtered,
            }

        return payload

    def _nearest_row(self, identity: str, target_timestamp: int) -> dict[str, Any] | None:
        rows = self._by_identity.get(identity)
        timestamps = self._timestamps_by_identity.get(identity)
        if not rows or not timestamps:
            return None

        idx = bisect.bisect_left(timestamps, target_timestamp)
        if idx == 0:
            return rows[0]
        if idx >= len(rows):
            return rows[-1]

        before = rows[idx - 1]
        after = rows[idx]
        if abs(before["timestamp"] - target_timestamp) <= abs(after["timestamp"] - target_timestamp):
            return before
        return after

    def _field_group_for_identity(self, identity: str) -> str:
        lowered = identity.strip().lower()
        if lowered in {"co_near", "cross_near"}: #, "co", "cross"}:
            return GROUP_NEAR_PAIR
        if lowered in {"co_far", "cross_far"}:
            return GROUP_FAR_PAIR
        if "raman" in lowered:
            return GROUP_RAMAN_PAIR
        if "far" in lowered:
            return GROUP_FAR_PAIR
        if "near" in lowered:
            return GROUP_NEAR_PAIR
        return GROUP_NEAR_PAIR

    def _nearest_group_timestamp(self, identities: list[str], target_timestamp: int) -> int | None:
        best_ts = None
        best_delta = None

        for identity in identities:
            timestamps = self._timestamps_by_identity.get(identity, [])
            if not timestamps:
                continue

            idx = bisect.bisect_left(timestamps, target_timestamp)
            candidates: list[int] = []
            if idx < len(timestamps):
                candidates.append(timestamps[idx])
            if idx > 0:
                candidates.append(timestamps[idx - 1])

            for candidate in candidates:
                delta = abs(candidate - target_timestamp)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_ts = candidate

        return best_ts

    def get_waveforms_payload(self, target_timestamp: int, sigma_v: float = 0, sigma_h: float = 0) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        records_filtered: list[dict[str, Any]] = []

        near_pair_identities = [i for i in self.identities if self._field_group_for_identity(i) == GROUP_NEAR_PAIR]
        far_pair_identities = [i for i in self.identities if self._field_group_for_identity(i) == GROUP_FAR_PAIR]
        raman_pair_identities = [i for i in self.identities if self._field_group_for_identity(i) == GROUP_RAMAN_PAIR]

        t0_anchor = self._nearest_group_timestamp(near_pair_identities, target_timestamp)
        if t0_anchor is None:
            t0_anchor = target_timestamp

        t1_target = t0_anchor + PULSE_OFFSET_MS
        t1_anchor = self._nearest_group_timestamp(far_pair_identities, t1_target)
        if t1_anchor is None:
            t1_anchor = t1_target

        t2_target = t0_anchor + (2 * PULSE_OFFSET_MS)
        t2_anchor = self._nearest_group_timestamp(raman_pair_identities, t2_target)
        if t2_anchor is None:
            t2_anchor = t2_target

        group_targets: dict[str, int] = {
            GROUP_NEAR_PAIR: t0_anchor,
            GROUP_FAR_PAIR: t1_anchor,
            GROUP_RAMAN_PAIR: t2_anchor,
        }
        group_offsets: dict[str, int] = {
            GROUP_NEAR_PAIR: 0,
            GROUP_FAR_PAIR: PULSE_OFFSET_MS,
            GROUP_RAMAN_PAIR: 2 * PULSE_OFFSET_MS,
        }

        # If filtering is requested, pre-compute filtered heatmaps for all identities
        filtered_heatmaps: dict[str, list[list[float]]] = {}
        if sigma_v > 0.01 or sigma_h > 0.01:
            for identity in self.identities:
                rows = self._by_identity.get(identity, [])
                timestamps = [r["timestamp"] for r in rows]
                waveforms = [r["waveform"] for r in rows]
                z_matrix = [
                    [waveform[bin_index] for waveform in waveforms]
                    for bin_index in range(self.samples_per_pulse)
                ]
                filtered_heatmaps[identity] = apply_gaussian_filter_2d(z_matrix, sigma_v, sigma_h)

        for identity in self.identities:
            group_name = self._field_group_for_identity(identity)
            group_target = group_targets[group_name]

            row = self._nearest_row(identity, group_target)
            if not row:
                continue

            records.append(
                {
                    "identity": identity,
                    "timestamp": row["timestamp"],
                    "delta_ms": group_offsets[group_name],
                    "tilt_deg": row["tilt_deg"],
                    "waveform": row["waveform"],
                }
            )
            
            # Extract filtered waveform if filtering is enabled
            if sigma_v > 0.01 or sigma_h > 0.01:
                rows = self._by_identity.get(identity, [])
                timestamps = [r["timestamp"] for r in rows]
                try:
                    ts_index = timestamps.index(row["timestamp"])
                    filtered_z_matrix = filtered_heatmaps[identity]
                    filtered_waveform = [filtered_z_matrix[bin_idx][ts_index] for bin_idx in range(self.samples_per_pulse)]
                    records_filtered.append(
                        {
                            "identity": identity,
                            "timestamp": row["timestamp"],
                            "delta_ms": group_offsets[group_name],
                            "tilt_deg": row["tilt_deg"],
                            "waveform": filtered_waveform,
                        }
                    )
                except (ValueError, IndexError):
                    # Fallback to raw waveform if extraction fails
                    records_filtered.append(records[-1])
            else:
                records_filtered.append(records[-1])

        return {
            "requested_timestamp": target_timestamp,
            "anchors": {
                "t0": t0_anchor,
                "t1": t1_anchor,
                "t2": t2_anchor,
                "pulse_offset_ms": PULSE_OFFSET_MS,
            },
            "bins": list(range(self.samples_per_pulse)),
            "waveforms": records,
            "waveforms_filtered": records_filtered,
        }

    def get_power_payload(self, identity: str | None = None) -> dict[str, Any]:
        if not self.identities:
            return {
                "identity": None,
                "identities": [],
                "timestamps": [],
                "power_format": "unknown",
                "power_a_min": [],
                "power_a_max": [],
                "power_b_min": [],
                "power_b_max": [],
                "power_a_sum": [],
                "power_b_sum": [],
                "ratio_ba_min": [],
                "ratio_ba_max": [],
            }

        # Collect all rows from all identities (no filtering)
        all_rows = []
        for identity_key in self.identities:
            all_rows.extend(self._by_identity.get(identity_key, []))

        def safe_float(value):
            """Convert value to float, return None if not finite or invalid."""
            try:
                f = float(value)
                return f if math.isfinite(f) else None
            except (ValueError, TypeError):
                return None

        timestamps = [int(row["timestamp"]) for row in all_rows]

        # Handle both old and new power formats
        if self.power_format == "old":
            power_a_min = [safe_float(row.get("power_a_min")) for row in all_rows]
            power_a_max = [safe_float(row.get("power_a_max")) for row in all_rows]
            power_b_min = [safe_float(row.get("power_b_min")) for row in all_rows]
            power_b_max = [safe_float(row.get("power_b_max")) for row in all_rows]

            ratio_ba_min = [
                _safe_ratio(b_min, a_min)
                for a_min, b_min in zip(power_a_min, power_b_min)
            ]
            ratio_ba_max = [
                _safe_ratio(b_max, a_max)
                for a_max, b_max in zip(power_a_max, power_b_max)
            ]

            return {
                "identity": "All",
                "identities": self.identities,
                "timestamps": timestamps,
                "power_format": "old",
                "power_a_min": power_a_min,
                "power_a_max": power_a_max,
                "power_b_min": power_b_min,
                "power_b_max": power_b_max,
                "power_a_sum": [],
                "power_b_sum": [],
                "ratio_ba_min": ratio_ba_min,
                "ratio_ba_max": ratio_ba_max,
            }
        else:  # new format
            power_a_sum = [safe_float(row.get("power_a_sum")) for row in all_rows]
            power_b_sum = [safe_float(row.get("power_b_sum")) for row in all_rows]

            ratio_ba_sum = [
                _safe_ratio(b_sum, a_sum)
                for a_sum, b_sum in zip(power_a_sum, power_b_sum)
            ]

            return {
                "identity": "All",
                "identities": self.identities,
                "timestamps": timestamps,
                "power_format": "new",
                "power_a_min": [],
                "power_a_max": [],
                "power_b_min": [],
                "power_b_max": [],
                "power_a_sum": power_a_sum,
                "power_b_sum": power_b_sum,
                "ratio_ba_min": ratio_ba_sum,
                "ratio_ba_max": [],
            }

    def get_laser_temp_payload(self, identity: str | None = None) -> dict[str, Any]:
        if not self.identities:
            return {
                "identity": None,
                "identities": [],
                "timestamps": [],
                "laser_temp_c": [],
            }

        selected_identity = (identity or "").strip()
        if selected_identity not in self._by_identity:
            selected_identity = self.identities[0]

        rows = self._by_identity.get(selected_identity, [])
        timestamps = [int(row["timestamp"]) for row in rows]
        laser_temp_c = [
            float(row["laser_temp_c"]) if math.isfinite(float(row["laser_temp_c"])) else None
            for row in rows
        ]

        return {
            "identity": selected_identity,
            "identities": self.identities,
            "timestamps": timestamps,
            "laser_temp_c": laser_temp_c,
        }

    def get_attitude_payload(self, identity: str | None = None) -> dict[str, Any]:
        if not self.identities:
            return {
                "identity": None,
                "identities": [],
                "timestamps": [],
                "roll_deg": [],
                "pitch_deg": [],
                "yaw_deg": [],
                "tilt_deg": [],
            }

        selected_identity = (identity or "").strip()
        if selected_identity not in self._by_identity:
            selected_identity = self.identities[0]

        rows = self._by_identity.get(selected_identity, [])
        timestamps = [int(row["timestamp"]) for row in rows]

        def _finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        roll_deg = [_finite_or_none(row["roll"]) for row in rows]
        pitch_deg = [_finite_or_none(row["pitch"]) for row in rows]
        yaw_deg = [_finite_or_none(row["yaw"]) for row in rows]
        tilt_deg = [_finite_or_none(row["tilt_deg"]) for row in rows]

        return {
            "identity": selected_identity,
            "identities": self.identities,
            "timestamps": timestamps,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "tilt_deg": tilt_deg,
        }

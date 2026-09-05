"""
Data type definitions for each processing level.
Each level is designed to be exportable to dataframe/CSV format.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class L0Data:
    """
    Raw ingested data (L0).
    - Preserves original ADC counts
    - Contains timing, metadata, no PM enrichment
    - No modifications applied
    """
    timestamp_utc: datetime
    sequence_id: int
    
    # Six channels: NF_CO, NF_CROSS, NF_RAMAN, FF_CO, FF_CROSS, FF_RAMAN
    # Each is raw ADC counts (int16 or int32 depending on digitizer)
    adc_counts: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    # e.g., {"pmt_hv": {...}, "adc_range": {...}, "laser_temp": 25.3, ...}
    
    # Pulse timing and trigger info
    trigger_time_ns: Optional[float] = None
    pulse_energy_monitor_raw: Optional[float] = None  # Pre-match value if present
    
    def to_dict(self) -> Dict:
        """Export to dictionary (for CSV, HDF5, etc.)"""
        # Convert numpy arrays to lists for JSON serialization
        adc_counts_serializable = {}
        for channel_id, adc_array in self.adc_counts.items():
            if hasattr(adc_array, 'tolist'):  # numpy array
                adc_counts_serializable[channel_id] = adc_array.tolist()
            else:
                adc_counts_serializable[channel_id] = list(adc_array)
        
        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "sequence_id": self.sequence_id,
            "trigger_time_ns": self.trigger_time_ns,
            "pulse_energy_monitor_raw": self.pulse_energy_monitor_raw,
            "adc_counts": adc_counts_serializable,
            **self.metadata,
        }


@dataclass
class L0bData:
    """
    Ingested + Power-Matched data (L0b).
    - Extends L0 with matched or fallback PM data
    - Preserves original L0 (no modifications)
    - Includes match provenance (quality, source, traceability)
    """
    l0: L0Data
    
    # Power matching results
    power_a_sum: float  # Matched or fallback constant
    power_b_sum: float
    
    # Provenance flags
    pm_source: str  # 'matched' | 'constant' | 'missing'
    
    # Match metadata (present if pm_source == 'matched')
    match_quality: Optional[float] = None  # 0-1, correlation score
    time_offset_ns: Optional[float] = None  # Time offset applied in matching
    distance_matched_samples: Optional[int] = None  # How many samples matched
    pm_file_source: Optional[str] = None  # Path to PM file used
    
    # Fallback metadata (present if pm_source == 'constant')
    fallback_reason: Optional[str] = None  # Why constant was used
    
    def to_dict(self) -> Dict:
        """Export to dictionary with provenance"""
        d = self.l0.to_dict()
        d.update({
            "power_a_sum": self.power_a_sum,
            "power_b_sum": self.power_b_sum,
            "pm_source": self.pm_source,
            "match_quality": self.match_quality,
            "time_offset_ns": self.time_offset_ns,
            "distance_matched_samples": self.distance_matched_samples,
            "pm_file_source": self.pm_file_source,
            "fallback_reason": self.fallback_reason,
        })
        return d


@dataclass
class L1Data:
    """
    Engineering corrected waveforms (L1).
    - Per-shot basis
    - Trigger registered, background subtracted, energy normalized, gain normalized
    - Uncertainty tracked per component (photon, background, energy, gain)
    """
    l0b: L0bData  # Link back to source
    
    # Corrected signals (six channels)
    signal: Dict[str, np.ndarray] = field(default_factory=dict)  # S-hat (eqn 6): gain-normalized, final signal

    # S-tilde (eqn 5): pulse-energy-normalized but pre-gain-normalization.
    # Retained separately because L1CrossProcessor's cross-channel ratios
    # (section 6.3) deliberately use this instead of S-hat -- with S-hat, the
    # per-channel PMT gain terms only cancel algebraically when both channels
    # share identical generic ("datasheet") characterization at the same HV;
    # with per-unit ("instance") characterization they would not cancel.
    signal_tilde: Dict[str, np.ndarray] = field(default_factory=dict)

    # Range assigned to each bin (eqn 1). Fixed per channel unless t0/tau_j are
    # recalibrated; computed here so downstream consumers never recompute it.
    range_m: Dict[str, np.ndarray] = field(default_factory=dict)

    # Component-wise uncertainties
    uncertainty_random: Dict[str, np.ndarray] = field(default_factory=dict)  # Photon + electronics
    uncertainty_background: Dict[str, np.ndarray] = field(default_factory=dict)
    uncertainty_energy: Dict[str, np.ndarray] = field(default_factory=dict)  # From pulse energy normalization
    uncertainty_gain: Dict[str, np.ndarray] = field(default_factory=dict)  # From calibration
    
    # Quality flags
    saturation_flags: Dict[str, np.ndarray] = field(default_factory=dict)  # Boolean per bin
    trigger_valid: bool = True

    # QA/QC tests (IOOS QARTOD flags, see processing.qa_flags.QartodFlag),
    # keyed by test name then channel_id. Not every test produces a scalar
    # value (some are flag-only) -- qa_values only has an entry when one
    # exists. qa_flags always has one per test that ran.
    # e.g. qa_values["adc_occupancy"]["co_near"] = Q_ADC array (eqn 9)
    #      qa_flags["adc_occupancy"]["co_near"] = QartodFlag array (int)
    qa_values: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    qa_flags: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # L2's K_lidar/c_est: per-shot scalars (not per-bin, unlike qa_values/
    # qa_flags above), so they get their own fields rather than folding into
    # those dicts. Only set for L2 rows on co_near/cross_near/co_far/cross_far
    # (see L2Processor._fit_k_lidar) -- None everywhere else, including plain
    # L1 rows (this dataclass is reused for L2's per-shot output -- see
    # L2Processor.process/app_v2.py's L2 route). qak_flag is a QartodFlag
    # int (1/3/4/9); None means "not attempted for this channel" (e.g.
    # raman_near, or an L1CrossProcessor ratio identity), distinct from 9
    # ("attempted, couldn't be computed").
    k_lidar: Optional[float] = None
    c_est: Optional[float] = None
    qak_flag: Optional[int] = None

    # In-situ ac-s IOP reference values: total beam attenuation ct(532nm)/
    # ct(650nm) -- the ac-s's own measured c(lambda) (particulate/CDOM
    # only, factory-calibrated against pure water) plus c_water_532/
    # c_water_650 added back, so these compare directly against K_lidar
    # (also a total-attenuation estimate) rather than c_est (K_lidar with
    # c_water already subtracted back out). Depth-averaged over the same
    # window that produced k_lidar/qak_flag above (nf/ff for the elastic
    # 532nm windows, raman_nf/raman_ff for the Raman-shifted 650nm windows
    # -- see iop_geo.profiles.OUTPUT_SPECS and app_v2.py's L2 route). Same
    # "L2-only, everywhere else None" status as k_lidar/c_est. None also
    # when no ac-s cast is contemporaneous with this shot (see
    # iop_geo.profiles.MATCH_PAD) or the matched cast has no data in that
    # depth window -- not just "not yet computed".
    ct532_near: Optional[float] = None
    ct532_far: Optional[float] = None
    ct650_near: Optional[float] = None
    ct650_far: Optional[float] = None

    # Processing metadata
    background_estimate: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (value, uncertainty) per channel
    processing_notes: str = ""
    
    def to_dataframe(self, channel: str, include_uncertainty=True) -> pd.DataFrame:
        """Export single channel to dataframe for CSV output"""
        data = {
            "timestamp_utc": self.l0b.l0.timestamp_utc,
            "sequence_id": self.l0b.l0.sequence_id,
            "range_m": self.range_m.get(channel, np.array([])),
            "signal": self.signal.get(channel, np.array([])),
            "saturation_flag": self.saturation_flags.get(channel, np.array([])),
        }
        if include_uncertainty:
            data.update({
                "unc_random": self.uncertainty_random.get(channel, np.array([])),
                "unc_background": self.uncertainty_background.get(channel, np.array([])),
                "unc_energy": self.uncertainty_energy.get(channel, np.array([])),
                "unc_gain": self.uncertainty_gain.get(channel, np.array([])),
            })
        for test_name, per_channel in self.qa_values.items():
            if channel in per_channel:
                data[f"qa_{test_name}"] = per_channel[channel]
        for test_name, per_channel in self.qa_flags.items():
            if channel in per_channel:
                data[f"qa_{test_name}_flag"] = per_channel[channel]
        return pd.DataFrame(data)


@dataclass
class L2Data:
    """
    Calibrated profiles (L2).
    - Ensemble-aware (uses multiple L1 shots for averages, overlap estimates, etc.)
    - Range-resolved, SNR computed, overlap/alignment diagnostics
    - Still contains per-shot + aggregated statistics
    """
    l0b_ensemble: List[L0bData]  # Source shots
    l1_ensemble: List[L1Data]  # Corrected shots

    # Per-shot range-corrected signal: X_j(r) = r^2 * S_hat_j, one dict per
    # shot in l1_ensemble (same order/indexing), keyed by that shot's
    # channel_id(s). Kept per-shot (not just the ensemble average below) so
    # individual shots can still be inspected/plotted.
    range_corrected: List[Dict[str, np.ndarray]] = field(default_factory=list)

    # Per-shot K_lidar/c_est/QAK, same one-dict-per-shot shape as
    # range_corrected above (same order/indexing into l1_ensemble), but only
    # keyed by channels the fit was attempted for (co_near/cross_near/
    # co_far/cross_far) -- see L2Processor._fit_k_lidar.
    k_lidar: List[Dict[str, Optional[float]]] = field(default_factory=list)
    c_est: List[Dict[str, Optional[float]]] = field(default_factory=list)
    qak_flag: List[Dict[str, int]] = field(default_factory=list)

    # Calibrated signals (averaged, range-corrected)
    signal_avg: Dict[str, np.ndarray] = field(default_factory=dict)  # Ensemble average
    signal_std: Dict[str, np.ndarray] = field(default_factory=dict)  # Standard deviation
    
    # Range axis
    range_bin: np.ndarray = field(default_factory=lambda: np.array([]))
    range_m: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # SNR and quality
    snr: Dict[str, np.ndarray] = field(default_factory=dict)  # Signal-to-noise ratio
    snr_threshold: float = 1.0  # Minimum SNR for valid data
    
    # Overlap estimates and diagnostics
    overlap_proxy_nf: Dict[str, np.ndarray] = field(default_factory=dict)  # NF overlap proxies per channel
    overlap_proxy_ff: Dict[str, np.ndarray] = field(default_factory=dict)  # FF overlap proxies
    
    # Saturation/linearity diagnostics
    linearity_ok: Dict[str, bool] = field(default_factory=dict)  # Per channel
    saturation_range_m: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (start, end) range
    
    # Alignment sensitivity (attitude response)
    attitude_sensitivity: Dict[str, float] = field(default_factory=dict)  # e.g., "NF_CO_roll_slope": 0.05
    
    # Raman diagnostics
    raman_leakage_flags: Dict[str, np.ndarray] = field(default_factory=dict)  # 532nm leakage detection
    
    processing_notes: str = ""
    
    def to_dataframe(self, channel: str) -> pd.DataFrame:
        """Export channel to dataframe"""
        return pd.DataFrame({
            "range_m": self.range_m,
            "signal_avg": self.signal_avg.get(channel, np.array([])),
            "signal_std": self.signal_std.get(channel, np.array([])),
            "snr": self.snr.get(channel, np.array([])),
            "saturation_flag": self.saturation_range_m.get(channel, (np.inf, -np.inf)),
        })


@dataclass
class L3Data:
    """
    Engineering products (L3).
    - Final stitched, depolarization-corrected, uncertainty-propagated products
    - Includes NF/FF scaling, blending, depolarization, Raman constraints
    - UNC-ready output structure (value, uncertainty, distribution, QC mask)
    """
    l2_input: L2Data  # Source L2 data
    
    # Stitched profiles (NF and FF blended)
    signal_stitched: Dict[str, np.ndarray] = field(default_factory=dict)  # Composite NF+FF
    blending_weight: np.ndarray = field(default_factory=lambda: np.array([]))  # Weighting function
    
    # NF/FF scaling factors
    nf_ff_scale: Dict[str, float] = field(default_factory=dict)  # Scale factor per channel family
    nf_ff_scale_uncertainty: Dict[str, float] = field(default_factory=dict)
    
    # Depolarization
    depolarization_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    depolarization_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    depolarization_bias: float = 0.0  # Instrument depolarization offset
    
    # Raman constraint
    raman_effective_attenuation: np.ndarray = field(default_factory=lambda: np.array([]))
    raman_atten_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Comprehensive uncertainty (UNC-ready)
    uncertainty_total: Dict[str, np.ndarray] = field(default_factory=dict)  # Combined standard uncertainty
    uncertainty_components: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # e.g., uncertainty_components["NF_CO"] = {"random": [...], "gain": [...], "overlap": [...], ...}
    
    # Quality/validity mask
    valid_data_mask: np.ndarray = field(default_factory=lambda: np.array([]))  # Boolean per bin
    qc_flags: Dict[str, np.ndarray] = field(default_factory=dict)  # Per-bin QC codes
    
    processing_notes: str = ""
    
    def to_dataframe(self, channel: str, include_unc_breakdown=False) -> pd.DataFrame:
        """Export to dataframe (NASA/UNC format ready)"""
        df = pd.DataFrame({
            "range_m": self.l2_input.range_m,
            "signal": self.signal_stitched.get(channel, np.array([])),
            "uncertainty": self.uncertainty_total.get(channel, np.array([])),
            "valid_flag": self.valid_data_mask,
        })
        if include_unc_breakdown and channel in self.uncertainty_components:
            for unc_type, unc_array in self.uncertainty_components[channel].items():
                df[f"unc_{unc_type}"] = unc_array
        return df

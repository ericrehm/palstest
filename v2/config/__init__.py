"""Configuration loading and schema validation"""
from pathlib import Path
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class L1Constants:
    """Global constants for L1 equation (1) range conversion and (5) energy
    normalization (see Appendix A 5.1/5.3). Per-channel tau_j lives in
    calibration.timing_offsets_ns; per-channel B_j/sigma_B_j live in
    calibration.background.
    """
    c0_m_per_s: float = 299792458.0   # vacuum speed of light
    n_w: float = 1.34                 # refractive index of seawater
    fs_hz: float = 2.5e9              # digitizer sampling rate
    t0_ns: float = 0.0                # global trigger/timing reference (lab-calibrated)
    e_ref_scale: float = 1.0


@dataclass
class ProcessingDefaults:
    """L1-L3 processing algorithm defaults"""
    background_mad_scale: float = 1.4826
    background_pre_trigger_bins: int = 50
    background_late_range_bins: int = 100
    energy_ref_percentile: int = 50
    adc_near_clipping_percent: float = 85.0
    adc_clipping_percent: float = 95.0
    snr_threshold_l2: float = 1.0
    g_adc_db: float = 0.0  # ADC/mux voltage-ratio loss, in dB (eqn 6)


@dataclass
class ChannelConfig:
    """Configuration for one receiver channel.

    pmt_gain_index says which "# pmt_gain_N" header line in a shot file holds
    this channel's actual per-run HV setpoint -- that voltage itself is
    operational data recorded per-file, not calibration data, so it is never
    read from here. See io.pals_io.parse_l0b_shots / _parse_pmt_gain_header.
    """
    channel_id: str
    pmt_model: str
    pmt_hv_nominal: float
    pmt_gain_index: int
    wavelength_nm: float
    adc_range_v: float
    adc_full_scale_code: int
    resistance_ohms: float
    ND_filter_OD: float = 0
    responsivity_file: str = None
    gain_file: str = None
    # Full raw JSON block for this channel, S_A/S_K and whatever future
    # per-unit parameters the "instance" PMT model needs -- deliberately not
    # promoted to typed fields above, so new instance-model parameters don't
    # require a code change here each time.
    raw: Optional[dict] = None


def _resolve_pmt_file(base_dir: Path, rel_path: str) -> str:
    """Resolve a responsivity/gain file path from PALS_SBS312.json. Paths are
    relative to the v2/ package directory (config_path.parent.parent), e.g.
    "../pmt/foo.csv" from v2/config/PALS_SBS312.json resolves to the
    project-root pmt/ directory."""
    if rel_path is None:
        return None
    return str((base_dir / rel_path).resolve())


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load instrument configuration from JSON file"""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        data = json.load(f)

    # v2/ package directory -- base for responsivity_file/gain_file paths
    base_dir = config_path.parent.parent

    config = {
        'instrument_id': data.get('instrument_id', 'PALS_UNKNOWN'),
        'channels': {},
        'calibration': data.get('calibration', {}),
        'l1_constants': L1Constants(**data.get('l1_constants', {})),
        'processing': ProcessingDefaults(**data.get('processing_defaults', {})),
        'metadata': data.get('metadata', {}),
    }

    for ch_id, ch_data in data.get('channels', {}).items():
        config['channels'][ch_id] = ChannelConfig(
            channel_id=ch_id,
            pmt_model=ch_data.get('pmt_model'),
            pmt_hv_nominal=ch_data.get('pmt_hv_nominal'),
            pmt_gain_index=ch_data.get('pmt_gain_index'),
            wavelength_nm=ch_data.get('wavelength_nm'),
            adc_range_v=ch_data.get('adc_range_v'),
            adc_full_scale_code=ch_data.get('adc_full_scale_code'),
            resistance_ohms=ch_data.get('resistance_ohms'),
            responsivity_file=_resolve_pmt_file(base_dir, ch_data.get('responsivity_file')),
            gain_file=_resolve_pmt_file(base_dir, ch_data.get('gain_file')),
            ND_filter_OD=ch_data.get('ND_filter_OD'),
            raw=ch_data,
        )

    return config


__all__ = ['load_config', 'ChannelConfig', 'ProcessingDefaults', 'L1Constants']

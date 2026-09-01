"""
Calibration and characterization classes for PALS instrument components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import numpy as np

# r9880U.py lives at the project root, alongside v2/ (same pattern pmt.py uses).
try:
    from r9880U import R9880UResponsivity, R9880UGain
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from r9880U import R9880UResponsivity, R9880UGain


# Unused: nothing in the codebase calls ADCCalibration, ND1Correction,
# PMTBlank, or ChannelCalibration below (only Responsivity/Gain, used by
# L1Processor, are live) -- eqn 4/6's ADC/gain handling ended up as direct
# computation in l1_processor.py instead of going through these composite
# classes. Left in place, commented out, as a record of the originally
# planned design rather than deleted outright.
#
# @dataclass
# class ADCCalibration:
#     """
#     Analog-to-Digital Converter calibration.
#     Converts raw ADC codes to physical units (volts).
#     """
#     channel_id: str
#     full_scale_code: int  # e.g., 65535 for 16-bit
#     adc_range_v: float  # Full-scale voltage range
#     offset_code: int  # Baseline offset (e.g., 32768 for bipolar)
#     adc_gain_v_per_code: Optional[float] = None  # Computed if not provided
#
#     def __post_init__(self):
#         if self.adc_gain_v_per_code is None:
#             self.adc_gain_v_per_code = self.adc_range_v / self.full_scale_code
#
#     def code_to_volts(self, adc_codes: np.ndarray) -> np.ndarray:
#         """Convert raw ADC codes to voltage"""
#         return (adc_codes - self.offset_code) * self.adc_gain_v_per_code
#
#     def volts_to_code(self, volts: np.ndarray) -> np.ndarray:
#         """Convert voltage back to ADC codes"""
#         return (volts / self.adc_gain_v_per_code) + self.offset_code
#
#     def is_clipped(self, adc_codes: np.ndarray, threshold_percent: float = 95.0) -> np.ndarray:
#         """
#         Detect clipped samples (ADC saturation).
#
#         Args:
#             adc_codes: Raw ADC array
#             threshold_percent: Percentage of full scale to flag as clipped
#
#         Returns:
#             Boolean array, True where clipped
#         """
#         clip_level = self.full_scale_code * (threshold_percent / 100.0)
#         return np.abs(adc_codes - self.offset_code) > clip_level
#
#
# @dataclass
# class ND1Correction:
#     """
#     Neutral Density filter correction (trivial, but explicit).
#     ND1 is a fixed attenuator in the Power A/B monitor path.
#     """
#     attenuation_factor: float  # e.g., 0.001 for ND1 ~3OD
#     name: str = "ND1"
#
#     def apply(self, power_raw: float) -> float:
#         """Reverse ND1 attenuation"""
#         return power_raw / self.attenuation_factor
#
#
# @dataclass
# class PMTBlank:
#     """
#     Dark count / blank response for a single PMT at fixed high voltage.
#     Typically small but must be tracked separately per channel.
#     """
#     channel_id: str
#     blank_count_hz: float  # Dark count rate in Hz
#     hv_nominal: float  # High voltage setting (Volts)
#     temperature_c: Optional[float] = None  # Temperature when measured
#     data_file: Optional[str] = None  # Reference to calibration file
#
#     def blank_signal(self, integration_time_s: float) -> float:
#         """Estimate blank counts over integration period"""
#         return self.blank_count_hz * integration_time_s


class Responsivity:
    """
    PMT cathode responsivity (mA/W) as a function of wavelength.

    Two ways to build one, selected by which classmethod you call:

    - from_datasheet(): the "class-based" model -- a generic, digitized
      datasheet curve for a PMT *model* (e.g. R9880U20), shared across
      every physical unit of that model. Wraps R9880UResponsivity from
      r9880U.py (see pmt.py for the reference usage pattern).
    - from_channel_config(): the "instance-based" model -- one built per
      physical channel/tube from that channel's own metadata block in
      PALS_SBS312.json (six of these, one per Identity). The math is
      channel-specific and TBD.

    Both expose the same __call__(wavelength_nm, extrapolate=False) ->
    (responsivity_mA_per_W, sigma), so calling code doesn't need to care
    which one it has.
    """

    def __init__(self) -> None:
        self._datasheet_model: Optional[R9880UResponsivity] = None
        self._channel_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_datasheet(cls, csv_path, **kwargs) -> 'Responsivity':
        """Class-based: shared digitized-datasheet curve (see r9880U.py)."""
        obj = cls()
        obj._datasheet_model = R9880UResponsivity(csv_path, **kwargs)
        return obj

    @classmethod
    def from_channel_config(cls, channel_config: Dict[str, Any]) -> 'Responsivity':
        """Instance-based: per-channel parameters from PALS_SBS312.json."""
        obj = cls()
        obj._channel_config = channel_config
        return obj

    def __call__(self, wavelength_nm, extrapolate: bool = False):
        if self._datasheet_model is not None:
            return self._datasheet_model(wavelength_nm, extrapolate=extrapolate)

        
        if self._channel_config is not None:
            V_lambda = 0.88 # Use lumens to watts correction 532 nm for Raman scattering at 650 nm too.
            lumems_per_W = 683 * V_lambda
            R_K = self._channel_config.get("S_K", 1.0) / 1e6 * lumems_per_W   # 1e6 uA / A
            sigma_R_K = 0.1 * R_K  # Example: 10% relative uncertainty
            return R_K, sigma_R_K

            # raise NotImplementedError(
            #     "Per-channel responsivity model not yet implemented -- fill in "
            #     "the math in Responsivity.__call__ using self._channel_config."
            # )
        raise RuntimeError(
            "Responsivity model not initialized -- use from_datasheet() or "
            "from_channel_config()."
        )


class Gain:
    """
    PMT gain (electron multiplication factor) as a function of supply/control
    voltage. Same from_datasheet() / from_channel_config() selection as
    Responsivity -- see its docstring.
    """

    def __init__(self) -> None:
        self._datasheet_model: Optional[R9880UGain] = None
        self._channel_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_datasheet(cls, csv_path, **kwargs) -> 'Gain':
        """Class-based: shared digitized-datasheet curve (see r9880U.py)."""
        obj = cls()
        obj._datasheet_model = R9880UGain(csv_path, **kwargs)
        return obj

    @classmethod
    def from_channel_config(cls, channel_config: Dict[str, Any]) -> 'Gain':
        """
        Instance-based: per-channel parameters from PALS_SBS312.json.
        """
        obj = cls()
        obj._channel_config = channel_config
        return obj

    def __call__(self, voltage_V, extrapolate: bool = True):
        """
        Evaluate the PMT gain for the given supply/control voltage.
        """
        if self._datasheet_model is not None:
            return self._datasheet_model(voltage_V, extrapolate=extrapolate)

        # See: "PALS Near-Field and Far-Field Receiver Signal, Collected-Power
        # and Alignment Analysis", section 5. PMT Gain Extrapolation and 
        # section 7. Measured Electrical Signal and PMT-Inferred Optical Power.

        if self._channel_config is not None:
            S_A = self._channel_config.get('S_A')
            S_K = self._channel_config.get('S_K')
            G_1000 = S_A / (S_K * 1e-6)
            GV = G_1000 * (voltage_V / 1000.0)**8 
            sigma_GV = 0.1 * GV  # Example: 10% relative uncertainty
            return GV, sigma_GV
        
            # raise NotImplementedError(
            #     "Per-channel gain model not yet implemented -- fill in the "
            #     "math in Gain.__call__ using self._channel_config."
            # )
        raise RuntimeError(
            "Gain model not initialized -- use from_datasheet() or "
            "from_channel_config()."
        )


# class ChannelCalibration:
#     """
#     Composite calibration for one receiver channel.
#     Combines ADC, PMT responsivity/gain, blanks, and ND1 correction.
#     """
#     def __init__(
#         self,
#         channel_id: str,
#         adc: ADCCalibration,
#         pmt_responsivity: Optional[Responsivity] = None,
#         pmt_gain: Optional[Gain] = None,
#         pmt_blank: Optional[PMTBlank] = None,
#         nd1: Optional[ND1Correction] = None,
#         g_amp: float = 1.0,  # Amplifier gain (fixed, always 1 for now)
#         g_adc: float = 0.001,  # ADC transfer coefficient (V/code or equivalent)
#     ):
#         self.channel_id = channel_id
#         self.adc = adc
#         self.pmt_responsivity = pmt_responsivity
#         self.pmt_gain = pmt_gain
#         self.pmt_blank = pmt_blank
#         self.nd1 = nd1
#         self.g_amp = g_amp  # Amplifier gain (fixed at 1)
#         self.g_adc = g_adc  # ADC transfer coefficient
#
#     def adc_to_physical(
#         self,
#         adc_codes: np.ndarray,
#         hv_v: Optional[float] = None,
#         energy_norm: float = 1.0,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Convert raw ADC counts to physical units (photons or power).
#
#         Applies:
#         1. ADC offset and scaling
#         2. PMT gain correction (if available)
#         3. Energy normalization
#         4. Uncertainty propagation
#
#         Returns:
#             (signal_physical, uncertainty)
#         """
#         # ADC to volts
#         volts = self.adc.code_to_volts(adc_codes)
#
#         # Uncertainty from ADC quantization
#         unc_adc = self.adc.adc_gain_v_per_code / np.sqrt(12)  # Uniform quantization noise
#
#         # Apply PMT gain correction if available
#         if self.pmt_gain is not None and hv_v is not None:
#             pmt_gain_m, _ = self.pmt_gain(hv_v)
#             signal = volts / (pmt_gain_m * self.g_amp * self.g_adc)
#             # Uncertainty: propagate through gain
#             unc_gain = unc_adc / (pmt_gain_m * self.g_amp * self.g_adc)
#         else:
#             signal = volts / (self.g_amp * self.g_adc)
#             unc_gain = unc_adc / (self.g_amp * self.g_adc)
#
#         # Energy normalization
#         if energy_norm != 1.0:
#             signal = signal / energy_norm
#             unc_energy = signal * 0.05  # Assume 5% energy monitor uncertainty (TBD)
#         else:
#             unc_energy = np.zeros_like(signal)
#
#         # Combined uncertainty
#         unc_total = np.sqrt(unc_gain**2 + unc_energy**2)
#
#         return signal, unc_total

"""
L1 Processing: Engineering Corrected Waveforms
- Range conversion (eqn 1, section 5.1)
- Background subtraction (eqn 4, section 5.2 — B_j/sigma_B_j come from a
  separate characterization step and are read from config, not computed here)
- Pulse-energy normalization (eqn 5, section 5.3)
- Fixed-gain normalization (eqn 6, section 5.4) -- signal is S-hat
- ADC occupancy QA flag (eqn 9) -- see _adc_occupancy()
- Detector SNR: per-shot, range-resolved, from this shot's own pre-trigger
  noise floor on raw counts -- see _detector_snr()
"""

from typing import Dict, Optional, Tuple, Type, TypeVar
import numpy as np
from ..data_types import L0bData, L1Data
from ..calibration import Responsivity, Gain
from .qa_flags import QartodFlag

_ModelT = TypeVar('_ModelT', Responsivity, Gain)


class L1Processor:
    """Convert L0b shots to L1 (engineering corrected). Each L0b shot carries
    exactly one channel (the real file format is one row per channel per
    Timestamp), so processing is per-shot, per-channel."""

    def __init__(self, config, pmt_model_source: str = "datasheet", pre_trigger_bins: Optional[int] = None):
        """
        Args:
            config: Instrument configuration dict (from load_config)
            pmt_model_source: "datasheet" (class-based, generic digitized
                curve shared across every unit of a PMT model) or "instance"
                (per-channel, built from that channel's own config block --
                math TBD). A UI/CLI choice, not a config-file setting: the
                same instrument config can be run either way.
            pre_trigger_bins: N for the per-shot detector SNR (see
                _detector_snr). A UI/CLI choice like pmt_model_source, not
                fixed by config -- defaults to
                processing_defaults.background_pre_trigger_bins when None.
                0 explicitly means "don't compute" -> NaN-filled.
        """
        if pmt_model_source not in ("datasheet", "instance"):
            raise ValueError(f"pmt_model_source must be 'datasheet' or 'instance', got {pmt_model_source!r}")

        self.config = config
        self.pmt_model_source = pmt_model_source
        self.l1_constants = config['l1_constants']
        self.channels = config['channels']
        self.tau_j_ns = config['calibration'].get('timing_offsets_ns', {})
        self.background = config['calibration'].get('background', {})
        self.processing_defaults = config['processing']
        self.pre_trigger_bins = (
            pre_trigger_bins if pre_trigger_bins is not None
            else self.processing_defaults.background_pre_trigger_bins
        )
        self._range_axis_cache: Dict[Tuple[str, int], np.ndarray] = {}

        # Build PMT responsivity/gain models once per channel (not per shot).
        # In "datasheet" mode, channels sharing the same digitized CSV share
        # one model instance rather than re-parsing the file per channel.
        self.responsivity_models: Dict[str, Optional[Responsivity]] = {}
        self.gain_models: Dict[str, Optional[Gain]] = {}
        self._datasheet_cache: Dict[Tuple[type, str], object] = {}
        for channel_id, ch_config in self.channels.items():
            self.responsivity_models[channel_id] = self._build_model(
                Responsivity, ch_config, 'responsivity_file'
            )
            self.gain_models[channel_id] = self._build_model(
                Gain, ch_config, 'gain_file'
            )

    def _build_model(self, model_cls: Type[_ModelT], ch_config, file_attr: str) -> Optional[_ModelT]:
        """Build one channel's responsivity/gain model per self.pmt_model_source.

        "instance" mode always builds (per-channel config always exists, even
        for channels with no digitized file, e.g. raman's null gain_file).
        "datasheet" mode requires a file and caches by path so channels
        sharing one CSV reuse a single parsed instance.
        """
        if self.pmt_model_source == "instance":
            return model_cls.from_channel_config(ch_config.raw)

        csv_path = getattr(ch_config, file_attr)
        if csv_path is None:
            return None
        key = (model_cls, csv_path)
        if key not in self._datasheet_cache:
            self._datasheet_cache[key] = model_cls.from_datasheet(csv_path)
        return self._datasheet_cache[key]  # type: ignore[return-value]

    def process(self, l0b: L0bData, e_ref: float) -> L1Data:
        """
        Apply L1 corrections to one L0b shot.

        Args:
            l0b: one shot's L0b data (single channel)
            e_ref: reference pulse energy for the file — mean PowerASum across
                all shots in the file (eqn 5's E_ref). Computed once per file
                by the caller and threaded through, since it isn't derivable
                from a single shot alone.

        Returns:
            L1Data with S-hat signal, range_m, background uncertainty, and
            QA values/flags (currently: eqn 9 ADC occupancy).
        """
        signal: Dict[str, np.ndarray] = {}
        signal_tilde: Dict[str, np.ndarray] = {}
        range_m: Dict[str, np.ndarray] = {}
        uncertainty_random: Dict[str, np.ndarray] = {}
        uncertainty_background: Dict[str, np.ndarray] = {}
        uncertainty_energy: Dict[str, np.ndarray] = {}
        uncertainty_gain: Dict[str, np.ndarray] = {}
        saturation_flags: Dict[str, np.ndarray] = {}
        background_estimate: Dict[str, Tuple[float, float]] = {}
        qa_values: Dict[str, Dict[str, np.ndarray]] = {"adc_occupancy": {}, "detector_snr": {}}
        qa_flags: Dict[str, Dict[str, np.ndarray]] = {"adc_occupancy": {}}

        # Power measured for this shot reflected by the wedge
        e_l = l0b.power_a_sum

        for channel_id, adc_array in l0b.l0.adc_counts.items():
            s = np.asarray(adc_array, dtype=float)
            n_bins = len(s)

            # Eqn 4: background subtraction. B_j/sigma_B_j are characterized
            # offline (blank files) and read from config, not computed here. (Eqns 2,3)
            b_j, sigma_b_j = self._background_for(channel_id)
            s_bg = s - b_j

            # Eqn 9: ADC occupancy QA flag. Must happen here, on the
            # blank-corrected raw counts in ADC-code units -- eqn 4b/5/6
            # below convert to volts and then physical units, so this is the
            # only level with access to this quantity.
            q_adc, q_adc_flag = self._adc_occupancy(s_bg, channel_id)
            qa_values["adc_occupancy"][channel_id] = q_adc
            qa_flags["adc_occupancy"][channel_id] = q_adc_flag

            # Detector SNR: per-shot, range-resolved, using this shot's own
            # pre-trigger noise floor on the RAW counts (not eqn 4's
            # blank-corrected s_bg) -- a self-contained diagnostic, no
            # calibration file or other shots needed.
            qa_values["detector_snr"][channel_id] = self._detector_snr(s)

            # Eqn 4b: convert counts to volts using adc_range_v and adc_full_scale_code
            adc_range_v = self.channels[channel_id].adc_range_v
            adc_full_scale_code = self.channels[channel_id].adc_full_scale_code
            s_bgv = s_bg * (adc_range_v / adc_full_scale_code)

            # Eqn 5: pulse-energy normalization
            #        S-tilde = (S - B_j) * (e_ref / e_l) * e_ref_scale
            # e_l (pulse energy for this shot), 
            # e_ref (average pulse energy across all shots)
            # e_ref_scale (scaling factor for reference energy, currently 1.0)
            # note that scaling is incomplete, as (S - Bj) is still in counts.
            e_ref_scale = self.l1_constants.e_ref_scale
            ratio = (e_ref / e_l) if e_l else 1.0
            s_tilde = s_bgv * ratio * e_ref_scale
            signal_tilde[channel_id] = s_tilde

            # Eqn. 6 Fixed-gain normalization
            wavelength_nm = self.channels[channel_id].wavelength_nm
            g_adc = 10 ** (self.processing_defaults.g_adc_db / 20.0)
            ohms = self.channels[channel_id].resistance_ohms    # i_a is the anode current in amperes

            # This shot's actual per-run PMT HV (operational data recorded in
            # the L0b header, not a config/calibration value -- see
            # config.pmt_gain_index and io.pals_io._parse_pmt_gain_header).
            hv_v = self._hv_for(channel_id, l0b)

            gain_model = self.gain_models[channel_id]
            resp_model = self.responsivity_models[channel_id]
            if gain_model is None or resp_model is None:
                missing = 'gain_file' if gain_model is None else 'responsivity_file'
                raise RuntimeError(
                    f"No {missing} configured for channel '{channel_id}' in "
                    f"pmt_model_source='datasheet' mode -- either add a digitized "
                    f"CSV for it in PALS_SBS312.json, or switch pmt_model_source "
                    f"to 'instance'."
                )
            g_pmt, sigma_g_pmt = gain_model(hv_v)
            R_pmt, sigma_R_pmt = resp_model(wavelength_nm)
            nd_filter_od = self.channels[channel_id].ND_filter_OD or 0
            a_nd = 10 ** (-nd_filter_od)
            s_hat = s_tilde / (g_adc * g_pmt * R_pmt * ohms * a_nd)

            signal[channel_id] = s_hat
            range_m[channel_id] = self._range_axis(channel_id, n_bins)
            uncertainty_background[channel_id] = np.full(n_bins, sigma_b_j)
            uncertainty_random[channel_id] = np.zeros(n_bins)  # not yet characterized
            uncertainty_energy[channel_id] = np.zeros(n_bins)  # not yet characterized
            uncertainty_gain[channel_id] = np.zeros(n_bins)    # eqn 6 not yet implemented
            saturation_flags[channel_id] = self._check_saturation(s, channel_id)
            background_estimate[channel_id] = (b_j, sigma_b_j)

        return L1Data(
            l0b=l0b,
            signal=signal,
            signal_tilde=signal_tilde,
            range_m=range_m,
            uncertainty_random=uncertainty_random,
            uncertainty_background=uncertainty_background,
            uncertainty_energy=uncertainty_energy,
            uncertainty_gain=uncertainty_gain,
            saturation_flags=saturation_flags,
            background_estimate=background_estimate,
            qa_values=qa_values,
            qa_flags=qa_flags,
            processing_notes=(
                "L1: eqn 1 (range), eqn 4 (background subtraction, B_j from config), "
                "eqn 5 (energy normalization), eqn 6 (gain normalization), "
                "eqn 9 (ADC occupancy QA flag), detector SNR (per-shot, "
                f"pre_trigger_bins={self.pre_trigger_bins})."
            ),
        )

    def _background_for(self, channel_id: str) -> Tuple[float, float]:
        entry = self.background.get(channel_id, {})
        return float(entry.get('B_j', 0.0)), float(entry.get('sigma_B_j', 0.0))

    def _adc_occupancy(self, s_bg: np.ndarray, channel_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Eqn 9: Q_ADC(i,k) = (S(i,k) - B_j) / ADC_FS_j, per bin.

        s_bg is the blank-corrected raw count (eqn 4's S(i,k) - B_j), still in
        ADC-code units -- this must run before eqn 4b/5/6 convert to volts and
        then physical units, since L1 is the only level with access to this
        quantity.

        Flag thresholds per the doc text ("Flag q >= 0.95 as near-clipping and
        q >= 0.995 as clipped"), classified into IOOS QARTOD codes (see
        qa_flags.QartodFlag): GOOD below 0.95, SUSPECT (near-clipping) from
        0.95, FAIL (clipped) from 0.995. MISSING is assigned defensively for
        NaN bins; UNKNOWN should never occur here since Q_ADC is always
        computable from data L1 always has.
        """
        adc_full_scale = self.channels[channel_id].adc_full_scale_code
        q_adc = s_bg / adc_full_scale

        flags = np.full(q_adc.shape, QartodFlag.GOOD, dtype=np.int8)
        flags[np.isnan(q_adc)] = QartodFlag.MISSING
        flags[q_adc >= 0.95] = QartodFlag.SUSPECT
        flags[q_adc >= 0.995] = QartodFlag.FAIL

        return q_adc, flags

    def _detector_snr(self, x: np.ndarray) -> np.ndarray:
        """Per-shot, range-resolved detector SNR using this shot's own
        pre-trigger noise floor:

            SNR_shot(i) = (x(i) - mean(x[0:N])) / std(x[0:N])

        x is the RAW ADC counts (not eqn 4's blank-corrected s_bg) -- this is
        a self-contained per-shot diagnostic, independent of the B_j/sigma_B_j
        characterization eqn 2/3 read from config. N = self.pre_trigger_bins
        (a runtime option defaulting from
        processing_defaults.background_pre_trigger_bins; see __init__).
        N=0 explicitly means "don't compute" -> NaN-filled, not an error.

        Only ever called for real channels: L1CrossProcessor's synthetic
        ratio identities never pass through this method, since they're built
        from already-processed L1Data after the fact, not raw counts -- so
        no explicit exclusion is needed here.
        """
        n_bins = len(x)
        if self.pre_trigger_bins == 0:
            return np.full(n_bins, np.nan)

        n = min(self.pre_trigger_bins, n_bins)
        window = x[:n]
        e_mean = np.mean(window)
        sigma_e = np.std(window)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x - e_mean) / sigma_e

    def _hv_for(self, channel_id: str, l0b: L0bData) -> Optional[float]:
        """Look up this shot's actual PMT HV: config.pmt_gain_index says which
        '# pmt_gain_N' header line is this channel's, and the value itself
        comes from the L0b file's header (parsed once per file by
        io.pals_io.parse_l0b_shots), never from PALS_SBS312.json. Returns the
        magnitude in volts -- recorded HV is negative, but datasheet gain
        curves (see r9880U.py) are parameterized by positive supply voltage.
        """
        gain_index = self.channels[channel_id].pmt_gain_index
        pmt_gain_by_index = l0b.l0.metadata.get('pmt_gain_by_index', {})
        raw_v = pmt_gain_by_index.get(gain_index)
        return abs(raw_v) if raw_v is not None else None

    def _range_axis(self, channel_id: str, n_bins: int) -> np.ndarray:
        """Eqn 1: r_i = c0 * (t_i - t0 - tau_j) / (2 * n_w), t_i = i / fs.
        Negative r_i is expected for bins captured before the pulse exits the
        instrument window. Fixed per (channel, n_bins) until config changes,
        so cached rather than recomputed per shot.
        """
        key = (channel_id, n_bins)
        if key not in self._range_axis_cache:
            c0 = self.l1_constants.c0_m_per_s
            n_w = self.l1_constants.n_w
            fs = self.l1_constants.fs_hz
            t0_s = self.l1_constants.t0_ns * 1e-9
            tau_j_s = self.tau_j_ns.get(channel_id, 0.0) * 1e-9

            i = np.arange(n_bins)
            t_i = i / fs
            self._range_axis_cache[key] = c0 * (t_i - t0_s - tau_j_s) / (2.0 * n_w)

        return self._range_axis_cache[key]

    def _check_saturation(self, adc_array: np.ndarray, channel_id: str) -> np.ndarray:
        """Flag samples at or above the channel's ADC full-scale code."""
        ch_config = self.channels.get(channel_id)
        full_scale = ch_config.adc_full_scale_code if ch_config else adc_array.max()
        return adc_array >= full_scale

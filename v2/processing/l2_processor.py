"""
L2 Processing: Calibrated Profiles
- Range correction and SNR estimation
- Overlap and alignment diagnostics
- Linearity and saturation testing
- Raman channel validation
"""

import dataclasses
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..data_types import L1Data, L2Data
from ..config import load_config
from .l1_cross_processor import L1CrossProcessor
from .qa_flags import QartodFlag

# Which fitting window (see L2Constants) applies to which channel. Raman
# gets its own nf/ff windows (different slope than the elastic co_/cross_
# channels -- see L2Constants); L1CrossProcessor's ratio identities aren't
# fit at all (see L2Processor._fit_k_lidar's callers in process()).
_NF_CHANNELS = frozenset({'co_near', 'cross_near'})
_FF_CHANNELS = frozenset({'co_far', 'cross_far'})
_RAMAN_NF_CHANNELS = frozenset({'raman_near'})
_RAMAN_FF_CHANNELS = frozenset({'raman_far'})


class L2Processor:
    """Convert ensemble of L1 shots to L2 (calibrated profiles)"""

    def __init__(
        self,
        config,
        fit_window_nf_m: Optional[Tuple[float, float]] = None,
        fit_window_ff_m: Optional[Tuple[float, float]] = None,
        fit_window_raman_nf_m: Optional[Tuple[float, float]] = None,
        fit_window_raman_ff_m: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            config: Instrument configuration dict
            fit_window_nf_m, fit_window_ff_m, fit_window_raman_nf_m,
                fit_window_raman_ff_m: per-request override of
                l2_constants.k_lidar_fit_window_{nf,ff,raman_nf,raman_ff}_m
                (see _fit_k_lidar) -- a UI/CLI choice like L1Processor's
                pmt_model_source/pre_trigger_bins, not a config-file setting.
                None (any of them) keeps the config default for that window.
        """
        self.config = config
        self.processing_defaults = config['processing']
        l2_constants = config['l2_constants']
        if any(w is not None for w in (
            fit_window_nf_m, fit_window_ff_m, fit_window_raman_nf_m, fit_window_raman_ff_m
        )):
            l2_constants = dataclasses.replace(
                l2_constants,
                k_lidar_fit_window_nf_m=fit_window_nf_m or l2_constants.k_lidar_fit_window_nf_m,
                k_lidar_fit_window_ff_m=fit_window_ff_m or l2_constants.k_lidar_fit_window_ff_m,
                k_lidar_fit_window_raman_nf_m=fit_window_raman_nf_m or l2_constants.k_lidar_fit_window_raman_nf_m,
                k_lidar_fit_window_raman_ff_m=fit_window_raman_ff_m or l2_constants.k_lidar_fit_window_raman_ff_m,
            )
        self.l2_constants = l2_constants
    
    def process(self, l1_ensemble: List[L1Data]) -> L2Data:
        """
        Process ensemble of L1 shots to produce L2 product.

        Steps:
        1. Range correction: X_j(r) = r^2 * S_hat_j, per shot (S_hat_j is
           L1's signal, r is L1's range_m -- both already computed per-channel
           in L1Processor, so this is just the elementwise product).
        1b. K_lidar/c_est: per-shot log-slope fit of X_j(r) over a fixed
            near-/far-field window (co_near/cross_near vs co_far/cross_far,
            or raman_near/raman_far's own separate window pair), see
            _fit_k_lidar.
        2. SNR estimation
        3. Overlap proxy estimation
        4. NF/FF ratio plateau assessment
        5. Linearity and saturation diagnostics
        6. Attitude sensitivity analysis (pitch/roll)

        Steps 3-6 are not yet implemented.

        Args:
            l1_ensemble: List of L1Data from consecutive shots (same channel)

        Returns:
            L2Data with per-shot range-corrected signal, per-shot K_lidar/
            c_est/QAK, ensemble average/std of the range-corrected signal,
            and SNR.
        """

        if not l1_ensemble:
            raise ValueError("Empty L1 ensemble")

        # Eqn 7: X_j(r) = r^2 * S_hat_j, per shot, per channel -- except for
        # L1CrossProcessor's ratio channels (qer_near/qer_far/depol_near/
        # depol_far). Those are already a ratio of two signals that would
        # each be range-corrected identically, so r^2 cancels out by
        # construction (r^2*S1 / r^2*S2 = S1/S2); re-applying it here would
        # reintroduce a spurious range dependence rather than cancel again.
        # K_lidar/c_est are fit alongside, on the same X_j, for the four
        # co-/cross-polarized channels and the two Raman channels (each
        # pair against its own nf/ff window, see _fit_window_for) -- only
        # the ratio identities above are skipped (see _fit_k_lidar's
        # docstring).
        range_corrected: List[Dict[str, np.ndarray]] = []
        k_lidar_by_shot: List[Dict[str, Optional[float]]] = []
        c_est_by_shot: List[Dict[str, Optional[float]]] = []
        qak_by_shot: List[Dict[str, int]] = []
        for l1 in l1_ensemble:
            rc_row: Dict[str, np.ndarray] = {}
            k_row: Dict[str, Optional[float]] = {}
            c_row: Dict[str, Optional[float]] = {}
            qak_row: Dict[str, int] = {}
            for channel_id, s_hat in l1.signal.items():
                if channel_id in L1CrossProcessor.RATIO_IDENTITIES:
                    rc_row[channel_id] = s_hat
                    continue
                range_m = l1.range_m[channel_id]
                x_j = (range_m ** 2) * s_hat
                rc_row[channel_id] = x_j

                window = self._fit_window_for(channel_id)
                if window is not None:
                    k_lidar, c_est, qak = self._fit_k_lidar(range_m, x_j, *window)
                    k_row[channel_id] = k_lidar
                    c_row[channel_id] = c_est
                    qak_row[channel_id] = qak
            range_corrected.append(rc_row)
            k_lidar_by_shot.append(k_row)
            c_est_by_shot.append(c_row)
            qak_by_shot.append(qak_row)

        # Ensemble average/std of the range-corrected signal (not raw S_hat --
        # this is what "L2 calibrated profile" means per the L2Data docstring).
        # Ratio channels (see above) can be nan (0/0) or inf (x/0) at a bin --
        # both mean "undefined here", so both are excluded via nan-aware
        # stats (np.nanmean/np.nanstd only skip nan on their own, not inf).
        # A bin where every shot is invalid legitimately has no valid
        # aggregate either; that expected "all-NaN slice" warning is
        # suppressed rather than the resulting nan values.
        signal_avg = {}
        signal_std = {}
        for channel_id in range_corrected[0].keys():
            channel_shots = np.array([rc[channel_id] for rc in range_corrected])
            channel_shots = np.where(np.isfinite(channel_shots), channel_shots, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                signal_avg[channel_id] = np.nanmean(channel_shots, axis=0)
                signal_std[channel_id] = np.nanstd(channel_shots, axis=0)

        # Range axis: copied straight from L1 (eqn 1's r_i), not recomputed --
        # L1Processor already built and cached it per channel. Fixed unless
        # t0/tau_j change, so every shot of this channel has the identical
        # array; just read it off the first shot.
        channel_id = next(iter(l1_ensemble[0].signal))
        range_m = l1_ensemble[0].range_m[channel_id]
        n_bins = len(range_m)

        # SNR estimation (placeholder)
        snr = {}
        for channel_id in signal_avg.keys():
            snr[channel_id] = signal_avg[channel_id] / (signal_std[channel_id] + 1e-6)

        return L2Data(
            l0b_ensemble=[l1.l0b for l1 in l1_ensemble],
            l1_ensemble=l1_ensemble,
            range_corrected=range_corrected,
            k_lidar=k_lidar_by_shot,
            c_est=c_est_by_shot,
            qak_flag=qak_by_shot,
            signal_avg=signal_avg,
            signal_std=signal_std,
            range_bin=np.arange(n_bins),
            range_m=range_m,
            snr=snr,
            processing_notes=(
                "L2: range correction (X_j = r^2 * S_hat_j; skipped for "
                "L1CrossProcessor's ratio channels, where r^2 already "
                "cancelled out), per-shot K_lidar/c_est/QAK for co_near/"
                "cross_near/co_far/cross_far/raman_near/raman_far (see "
                "_fit_k_lidar), and ensemble avg/std/SNR. Overlap, "
                "linearity, and attitude "
                "diagnostics not yet implemented."
            ),
        )

    def _fit_window_for(self, channel_id: str) -> Optional[Tuple[float, float]]:
        """(r1, r2) fitting window for this channel, or None if K_lidar isn't
        attempted for it (an L1CrossProcessor ratio identity)."""
        if channel_id in _NF_CHANNELS:
            return tuple(self.l2_constants.k_lidar_fit_window_nf_m)
        if channel_id in _FF_CHANNELS:
            return tuple(self.l2_constants.k_lidar_fit_window_ff_m)
        if channel_id in _RAMAN_NF_CHANNELS:
            return tuple(self.l2_constants.k_lidar_fit_window_raman_nf_m)
        if channel_id in _RAMAN_FF_CHANNELS:
            return tuple(self.l2_constants.k_lidar_fit_window_raman_ff_m)
        return None

    def _fit_k_lidar(
        self, range_m: np.ndarray, x_j: np.ndarray, r1: float, r2: float
    ) -> Tuple[Optional[float], Optional[float], int]:
        """K_lidar/c_est from a single shot's range-corrected signal X_j(r).

        Classic single-scattering elastic-lidar log-slope method: within
        [r1, r2], ln(X_j) vs r is linear with slope ~= -2*c (two-way
        attenuation), so K_lidar = -slope/2 and c_est = K_lidar + c_water
        (the total estimated attenuation, referenced back up from the
        lidar-derived value using clear-water's known baseline).

        X_j can be <= 0 in-window (background-subtracted noise), where
        ln() is undefined -- those bins are dropped from the fit rather
        than the whole shot. QAK follows r^2 of the fit: GOOD (>=0.95),
        SUSPECT (>=0.90), FAIL (<0.90); MISSING (couldn't fit at all --
        fewer than l2_constants.k_lidar_min_fit_points valid bins in
        [r1, r2], including an empty window) returns (None, None, MISSING).
        """
        in_window = (range_m >= r1) & (range_m <= r2)
        r_win = range_m[in_window]
        x_win = x_j[in_window]
        valid = np.isfinite(x_win) & (x_win > 0)
        r_valid = r_win[valid]

        if len(r_valid) < self.l2_constants.k_lidar_min_fit_points:
            return None, None, int(QartodFlag.MISSING)

        y_valid = np.log(x_win[valid])
        slope, intercept = np.polyfit(r_valid, y_valid, 1)
        y_pred = slope * r_valid + intercept
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

        k_lidar = -slope / 2.0

        # Subtract the clear-water baseline to estimate the c_pg like ac-s
        # (c_pg = c_total - c_water)
        c_est = k_lidar - self.l2_constants.c_water_532

        if r_squared >= 0.95:
            qak = QartodFlag.GOOD
        elif r_squared >= 0.90:
            qak = QartodFlag.SUSPECT
        else:
            qak = QartodFlag.FAIL

        return float(k_lidar), float(c_est), int(qak)

    def build_ensemble_snr(self, l1_ensemble: List[L1Data], channel_id: str) -> L1Data:
        """
        Ensemble SNR = mean(S-tilde) / std(S-tilde) across every same-identity
        shot in the file, per bin. A whole-file aggregate, not tied to any
        single shot, so this returns one new synthetic-Identity row per
        channel (named "<channel_id>_ensemble_snr") rather than a per-shot
        result -- anchored to the *first* shot's own Timestamp/metadata for
        traceability, matching L1CrossProcessor's synthetic-identity pattern
        so it can be written into the same Matrix-style output file and
        plotted like any other identity.

        Deliberately uses S-tilde, not S-hat or the range-corrected X_j:
        same reasoning as L1CrossProcessor's ratios -- a gain-model-dependent
        quantity would make this vary with the pmt_model_source choice,
        which isn't wanted for a noise/consistency diagnostic. Applies
        uniformly to L1CrossProcessor's ratio identities too (their
        signal_tilde is equally well-defined), not just the six real
        channels, unless that turns out to be unwanted.
        """
        stack = np.array([l1.signal_tilde[channel_id] for l1 in l1_ensemble])
        stack = np.where(np.isfinite(stack), stack, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ensemble_snr = mean / std

        result_identity = f"{channel_id}_ensemble_snr"
        anchor = l1_ensemble[0]
        anchor_l0 = anchor.l0b.l0
        synthetic_l0 = dataclasses.replace(
            anchor_l0,
            adc_counts={},
            metadata={**anchor_l0.metadata, 'identity': result_identity},
        )
        synthetic_l0b = dataclasses.replace(anchor.l0b, l0=synthetic_l0)

        return L1Data(
            l0b=synthetic_l0b,
            signal={result_identity: ensemble_snr},
            signal_tilde={result_identity: ensemble_snr},
            range_m={result_identity: anchor.range_m[channel_id]},
            processing_notes=(
                f"L2 ensemble SNR for {channel_id}: mean(S-tilde)/std(S-tilde) "
                f"across {len(l1_ensemble)} shots in the file."
            ),
        )
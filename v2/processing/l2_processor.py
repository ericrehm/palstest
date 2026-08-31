"""
L2 Processing: Calibrated Profiles
- Range correction and SNR estimation
- Overlap and alignment diagnostics
- Linearity and saturation testing
- Raman channel validation
"""

import dataclasses
import warnings
from typing import Dict, List
import numpy as np
from ..data_types import L1Data, L2Data
from ..config import load_config
from .l1_cross_processor import L1CrossProcessor


class L2Processor:
    """Convert ensemble of L1 shots to L2 (calibrated profiles)"""
    
    def __init__(self, config):
        """
        Args:
            config: Instrument configuration dict
        """
        self.config = config
        self.processing_defaults = config['processing']
    
    def process(self, l1_ensemble: List[L1Data]) -> L2Data:
        """
        Process ensemble of L1 shots to produce L2 product.

        Steps:
        1. Range correction: X_j(r) = r^2 * S_hat_j, per shot (S_hat_j is
           L1's signal, r is L1's range_m -- both already computed per-channel
           in L1Processor, so this is just the elementwise product).
        2. SNR estimation
        3. Overlap proxy estimation
        4. NF/FF ratio plateau assessment
        5. Linearity and saturation diagnostics
        6. Attitude sensitivity analysis (pitch/roll)

        Steps 3-6 are not yet implemented.

        Args:
            l1_ensemble: List of L1Data from consecutive shots (same channel)

        Returns:
            L2Data with per-shot range-corrected signal, ensemble
            average/std of it, and SNR.
        """

        if not l1_ensemble:
            raise ValueError("Empty L1 ensemble")

        # Eqn 7: X_j(r) = r^2 * S_hat_j, per shot, per channel -- except for
        # L1CrossProcessor's ratio channels (qer_near/qer_far/depol_near/
        # depol_far). Those are already a ratio of two signals that would
        # each be range-corrected identically, so r^2 cancels out by
        # construction (r^2*S1 / r^2*S2 = S1/S2); re-applying it here would
        # reintroduce a spurious range dependence rather than cancel again.
        range_corrected: List[Dict[str, np.ndarray]] = []
        for l1 in l1_ensemble:
            range_corrected.append({
                channel_id: (
                    s_hat if channel_id in L1CrossProcessor.RATIO_IDENTITIES
                    else (l1.range_m[channel_id] ** 2) * s_hat
                )
                for channel_id, s_hat in l1.signal.items()
            })

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
            signal_avg=signal_avg,
            signal_std=signal_std,
            range_bin=np.arange(n_bins),
            range_m=range_m,
            snr=snr,
            processing_notes=(
                "L2: range correction (X_j = r^2 * S_hat_j; skipped for "
                "L1CrossProcessor's ratio channels, where r^2 already "
                "cancelled out) and ensemble avg/std/SNR. Overlap, linearity, "
                "and attitude diagnostics not yet implemented."
            ),
        )

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
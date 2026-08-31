"""
L1CrossProcessor: cross-channel diagnostics needing two *different* channels'
L1 shots at once -- something L1Processor can't do, since it processes one
single-channel shot at a time (see l1_processor.py's module docstring).

Implements section 6.3, "Cross-channel saturation diagnostics (L1)":
    Q_ER,NF(r)   = NF_CO(r)    / NF_RAMAN(r)   (eqn 12a)
    Q_ER,FF(r)   = FF_CO(r)    / FF_RAMAN(r)   (eqn 12b)
    delta_NF,raw(r) = NF_CROSS(r) / NF_CO(r)   (eqn 13a)
    delta_FF,raw(r) = FF_CROSS(r) / FF_CO(r)   (eqn 13b)

This is "a bit of both" QA and science: a genuine measured quantity, but the
doc also warns "do not interpret depolarization where the co-polarized
channel is saturated or nonlinear" -- so each result carries a QA flag (eqn 9,
from L1Processor) alongside it for that purpose. That flag is the QARTOD
"worst-of" combination of *both* contributing channels' own Q_ADC_FLAG, not
just the co-channel's -- either the numerator or the denominator could be the
saturated one and corrupt the ratio (see _combine_flags).

Deliberately uses S-tilde (eqn 5's pulse-energy-normalized signal), not S-hat
(eqn 6's gain-normalized signal) -- see L1Data.signal_tilde's docstring for
why: S-hat's per-channel gain terms don't reliably cancel in a ratio.

Runs as a separate step *after* L1Processor has produced every shot in a
file: groups the resulting L1Data by channel, does the timestamp matching
described below, and emits new *synthetic-Identity* L1Data rows (qer_near,
qer_far, depol_near, depol_far) rather than adding columns to the real
channels' rows -- every row in the L1 output keeps the same schema
regardless of Identity (see the file-format discussion this followed).

Timestamp matching (confirmed against the instrument's actual round-robin
acquisition order):
- eqn 13a/13b: cross_near/co_near always share one Timestamp (same for
  cross_far/co_far), so this is an exact match.
- eqn 12a/12b: co_near and raman_near (co_far and raman_far) do NOT share
  timestamps. For each co shot, match the nearest *strictly later* Raman
  shot; only if none exists after it (tail of file) fall back to the nearest
  in either direction. The round-robin fires co_near/cross_near twice per
  cycle but raman_near/raman_far once, so this naturally produces two
  Q_ER,NF values per cycle with different match qualities (one tightly
  paired within the same cycle, one loosely paired to the next cycle's
  Raman shot) -- both are kept; best-effort nearest-neighbor is accepted.
"""

import dataclasses
from typing import Dict, List, Optional
import numpy as np
from ..data_types import L1Data


class L1CrossProcessor:
    """Compute section 6.3's cross-channel ratios over one file's L1 shots."""

    # (result_identity, numerator_channel, denominator_channel, match_mode, co_channel)
    # match_mode: "exact" (share a Timestamp) or "forward_nearest" (nearest
    # strictly-later shot on the denominator side, tail-of-file fallback to
    # nearest in either direction).
    _RATIOS = [
        ("depol_near", "cross_near", "co_near", "exact", "co_near"),
        ("depol_far", "cross_far", "co_far", "exact", "co_far"),
        ("qer_near", "co_near", "raman_near", "forward_nearest", "co_near"),
        ("qer_far", "co_far", "raman_far", "forward_nearest", "co_far"),
    ]

    #: Every synthetic Identity this class produces. A ratio of two signals
    #: that would each be range-corrected identically already has its r^2
    #: cancel out by construction -- L2Processor checks this set so it
    #: doesn't re-apply r^2 to these and reintroduce a spurious range
    #: dependence (see l2_processor.py).
    RATIO_IDENTITIES = frozenset(result_identity for result_identity, *_ in _RATIOS)

    def process(self, l1_shots: List[L1Data]) -> List[L1Data]:
        """Given every L1 shot from one file (all channels, unsorted), return
        the derived qer_near/qer_far/depol_near/depol_far shots."""
        by_channel: Dict[str, List[L1Data]] = {}
        for shot in l1_shots:
            channel_id = next(iter(shot.signal_tilde))
            by_channel.setdefault(channel_id, []).append(shot)
        for shots in by_channel.values():
            shots.sort(key=lambda s: s.l0b.l0.sequence_id)

        derived: List[L1Data] = []
        for result_identity, num_ch, den_ch, mode, co_ch in self._RATIOS:
            derived.extend(
                self._compute_ratio(by_channel, result_identity, num_ch, den_ch, mode, co_ch)
            )
        return derived

    def _compute_ratio(
        self, by_channel, result_identity: str, num_ch: str, den_ch: str,
        mode: str, co_ch: str,
    ) -> List[L1Data]:
        numerators = by_channel.get(num_ch, [])
        denominators = by_channel.get(den_ch, [])
        if not numerators or not denominators:
            return []

        results = []
        for num_shot in numerators:
            den_shot = self._match(num_shot, denominators, mode)
            if den_shot is None:
                continue

            # A zero-signal bin (common near r=0, pre-trigger) makes the ratio
            # inf/nan -- mathematically correct (undefined ratio), not an
            # error, so this suppresses the warning rather than the value.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = num_shot.signal_tilde[num_ch] / den_shot.signal_tilde[den_ch]

            # Either contributing channel could be saturated/suspect and
            # corrupt the ratio -- not just the co-polarized one the doc text
            # calls out -- so combine both channels' own Q_ADC_FLAG (eqn 9)
            # via QARTOD's standard "worst flag wins" precedence. QartodFlag's
            # codes are deliberately ordered so severity increases with the
            # integer value (GOOD=1 < UNKNOWN=2 < SUSPECT=3 < FAIL=4 <
            # MISSING=9), so an elementwise max() directly implements that
            # precedence -- no explicit precedence table needed.
            num_flag = num_shot.qa_flags.get("adc_occupancy", {}).get(num_ch)
            den_flag = den_shot.qa_flags.get("adc_occupancy", {}).get(den_ch)
            combined_flag = self._combine_flags(num_flag, den_flag)

            co_shot = num_shot if num_ch == co_ch else den_shot
            results.append(self._build_synthetic_shot(result_identity, co_shot, ratio, combined_flag))
        return results

    @staticmethod
    def _combine_flags(
        flag_a: Optional[np.ndarray], flag_b: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """QARTOD aggregate flag: worst-of, via elementwise max (see the
        comment at the call site for why max() implements the precedence)."""
        if flag_a is None:
            return flag_b
        if flag_b is None:
            return flag_a
        return np.maximum(flag_a, flag_b)

    def _match(self, num_shot: L1Data, candidates: List[L1Data], mode: str) -> Optional[L1Data]:
        t0 = num_shot.l0b.l0.sequence_id
        if mode == "exact":
            for c in candidates:
                if c.l0b.l0.sequence_id == t0:
                    return c
            return None

        # forward_nearest
        after = [c for c in candidates if c.l0b.l0.sequence_id > t0]
        if after:
            return min(after, key=lambda c: c.l0b.l0.sequence_id)
        return min(candidates, key=lambda c: abs(c.l0b.l0.sequence_id - t0))

    def _build_synthetic_shot(
        self, result_identity: str, co_shot: L1Data, ratio: np.ndarray,
        co_qadc_flag: Optional[np.ndarray],
    ) -> L1Data:
        """Anchor the derived row to co_shot: its own Timestamp/Roll/Pitch/
        Yaw/PowerASum/PowerBSum, since that's the channel this diagnostic is
        about (both for the flag-borrowing reason above, and because it's
        always one of the two channels feeding the ratio)."""
        co_channel = next(iter(co_shot.signal_tilde))
        anchor_l0 = co_shot.l0b.l0
        synthetic_l0 = dataclasses.replace(
            anchor_l0,
            adc_counts={},  # no raw counts for a derived quantity
            metadata={**anchor_l0.metadata, 'identity': result_identity},
        )
        synthetic_l0b = dataclasses.replace(co_shot.l0b, l0=synthetic_l0)

        qa_flags: Dict[str, Dict[str, np.ndarray]] = {}
        if co_qadc_flag is not None:
            # Same test name ("adc_occupancy") as the real channels, so the
            # output CSV's QADCFLAG_ columns line up across every row type.
            qa_flags['adc_occupancy'] = {result_identity: co_qadc_flag}

        return L1Data(
            l0b=synthetic_l0b,
            signal={result_identity: ratio},
            signal_tilde={result_identity: ratio},
            range_m={result_identity: co_shot.range_m[co_channel]},
            qa_flags=qa_flags,
            processing_notes=(
                f"L1 cross-channel diagnostic: {result_identity} "
                f"(section 6.3, eqn 12/13), computed from S-tilde."
            ),
        )

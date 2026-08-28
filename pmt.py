
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from r9880U import R9880UResponsivity, R9880UGain
except Exception:
    try:
        from .r9880U import R9880UResponsivity, R9880UGain
    except Exception:
        R9880UResponsivity = None
        R9880UGain = None

@dataclass
class PMTCal:
    resp_model: any
    gain_model: any
    resistance_ohms: float

    @classmethod
    def from_files(cls, root_dir: str | Path, rel_pmt_dir: str | Path,
                   resp_csv: str, gain_csv: str, resistance_ohms: float) -> 'PMTCal':
        root_dir = Path(root_dir)
        pmt_dir = root_dir / rel_pmt_dir
        resp_path = pmt_dir / resp_csv
        gain_path = pmt_dir / gain_csv
        if R9880UResponsivity is None or R9880UGain is None:
            raise ImportError("r9880U.py not found on PYTHONPATH. Place r9880U20.py in repo root or install it.")
        resp_model = R9880UResponsivity(resp_path)
        gain_model = R9880UGain(gain_path)
        return cls(resp_model=resp_model, gain_model=gain_model, resistance_ohms=float(resistance_ohms))

    @classmethod
    def from_manifest_row(cls, row: dict) -> 'PMTCal':
        """Construct PMTCal from a manifest/ExpGrid row dict."""
        root_dir = row.get('root_dir')
        rel = row.get('pmt_rel_path') or 'data/PMT'
        resp_csv = row.get('pmt_resp_file') or 'R9880U20_responsivity_digitized.csv'
        gain_csv = row.get('pmt_gain_file') or 'R9880U20_gain_digitized.csv'
        resistance = row.get('pmt_resistance_ohms') or 50.0
        return cls.from_files(root_dir=root_dir, rel_pmt_dir=rel, resp_csv=resp_csv, gain_csv=gain_csv, resistance_ohms=resistance)

    @classmethod
    def from_expgrid(cls, meta: dict, PMT: str) -> 'PMTCal':
        """Construct PMTCal from a raw ExpGrid."""
        root_dir = meta.get('rootDir')
        pmts = meta.get('pmts')
        rel = pmts.get('relative_path_to_PMT_files') or '../PMT'
        pmt = pmts.get(PMT)
        if pmt is None:
            raise ValueError(f"PMT '{PMT}' not found in ExpGrid pmts section.") 
        resp_csv = pmt.get('responsivity').get('file') or 'R9880U20_responsivity_digitized.csv'
        gain_csv = pmt.get('gain').get('file') or 'R9880U20_gain_digitized.csv'
        resistance = pmt.get('resistance_ohms') or 50.0
        return cls.from_files(root_dir=root_dir, rel_pmt_dir=rel, resp_csv=resp_csv, gain_csv=gain_csv, resistance_ohms=resistance)

    def volts_to_watts(self, volts: np.ndarray, wavelength_nm: float, control_voltage_V: float,
                        supply_multiplier: float |  None = None) -> np.ndarray:
        resp_mA_per_W, _ = self.resp_model(wavelength_nm)

        if supply_multiplier is not None:
            gain, _ = self.gain_model(control_voltage_V * supply_multiplier)
        else:
            gain, _ = self.gain_model(control_voltage_V)
        G = self.resistance_ohms * (resp_mA_per_W / 1000.0) * gain
        G = max(G, 1e-24)
        # print(f"Gain G = {G:.3e}")
        return volts / G

# PALS V2 Processing Architecture

## Overview

A modular, uncertainty-aware processing pipeline for PALS lidar data:

```
L0 (Raw)
  ↓
[L0b Matcher] ← PM data enrichment
  ↓
L0b (Matched)
  ↓
[L1 Processor] ← Trigger reg, background, energy norm, gain norm
  ↓
L1 (Engineering Corrected)
  ↓
[L2 Processor] ← Ensemble average, range correction, overlap, linearity
  ↓
L2 (Calibrated Profiles)
  ↓
[L3 Processor] ← NF/FF stitching, depolarization, Raman attenuation, QC
  ↓
L3 (Final Products, UNC-ready)
```

## Directory Structure

```
v2/
├── __init__.py              # Package init, exports main classes
├── data_types.py            # Data structures: L0Data, L0bData, L1Data, L2Data, L3Data
├── calibration/
│   ├── __init__.py
│   └── classes.py           # ADC, ND1, PMT, Blank, ChannelCalibration
├── config/
│   ├── __init__.py
│   └── PALS_SBS312.json     # Instrument-specific calibration file
├── io/
│   ├── __init__.py
│   └── pals_io.py           # Read L0, write L1-L3 to CSV/HDF5
├── processing/
│   ├── __init__.py
│   ├── l0b_matcher.py       # Wrapper around match_power_data.py
│   ├── l1_processor.py      # Background, energy norm, gain norm
│   ├── l2_processor.py      # Overlap, linearity, SNR, diagnostics
│   └── l3_processor.py      # Stitching, depolarization, uncertainty
├── tests/
│   └── (placeholder for unit tests)
├── output/
│   └── (where CSV/HDF5 products are written)
├── app_v2.py                # Flask wrapper (thin client)
└── README.md                # This file
```

## Key Features

### Data Structures (types.py)

Each processing level is a dataclass with:
- **Core data** (numpy arrays): signals, uncertainties, flags
- **Metadata**: timestamps, configuration, provenance
- **Export methods**: `.to_dataframe()`, `.to_dict()` for CSV/JSON output

**L0Data**: Raw ADC counts, no modifications
**L0bData**: L0 + matched PM data with provenance (match quality, source, fallback flags)
**L1Data**: Corrected signals (background subtracted, energy normalized, gain normalized) + component uncertainties
**L2Data**: Ensemble averages, range-corrected signals, SNR, overlap proxies, linearity diagnostics
**L3Data**: Stitched NF/FF profiles, depolarization, Raman constraints, UNC-ready uncertainty breakdown

### Calibration (calibration/)

**Classes**:
- `ADCCalibration`: ADC offset, full-scale code, clipping detection
- `ND1Correction`: Neutral density filter (trivial attenuation factor)
- `PMTBlank`: Dark count rates (Hz)
- `PMTResponsivity`: Responsivity (A/W) vs HV from lab data files
- `PMTGain`: Gain vs HV from lab data files
- `ChannelCalibration`: Composite (ADC + PMT + ND1)

All feed into `adc_to_physical()` method for unit conversion with uncertainty propagation.

### Configuration (config/)

**Single JSON per instrument** (e.g., `PALS_SBS312.json`):
- Channel definitions (PMT models, HV, ADC range)
- Calibration constants (timing offsets, ADC offsets, blank counts)
- File paths to responsivity/gain data (pmt.py, r9880U.py)
- Processing algorithm defaults (background thresholds, SNR thresholds, etc.)

Loaded at startup; can be hot-reloaded for sensitivity studies.

### I/O (io/)

**Input**:
- `read_l0_from_csv(path)`: Parse shot CSV files

**Output** (exportable at each level):
- `write_l0_to_csv()`, `write_l1_to_csv()`, `write_l2_to_csv()`, `write_l3_to_csv()`
- Each converts internal numpy arrays → pandas DataFrame → CSV
- `write_l1_to_hdf5()`: Efficient binary storage option

**Format**: Each CSV row is one range bin; columns include signal, uncertainty, QC flags
**UNC-ready**: L3 output includes uncertainty component breakdown (photon, background, gain, overlap, etc.)

### Processing (processing/)

**L0bMatcher**: 
- Wraps `match_power_data.py` to enrich L0 with PM data
- Tracks provenance: match quality, time offset, source file
- Graceful fallback to constant if no match found

**L1Processor**:
- Trigger registration (timing alignment)
- Background subtraction (pre-trigger + late-range, MAD-based robust estimator)
- Pulse energy normalization (divide by Power A monitor)
- Fixed-gain normalization (ADC → physical units via calibration)
- Component-wise uncertainty tracking

**L2Processor**:
- Ensemble averaging (multiple L1 shots)
- Range correction (signal × (r/r_ref)²)
- SNR estimation
- Overlap proxy estimation (NF/FF ratio plateaus)
- Linearity/saturation diagnostics
- Attitude sensitivity (pitch/roll regression)

**L3Processor**:
- NF/FF scaling factors (robust regression, inverse variance weighting)
- Blending function (cosine/logistic transition)
- Stitched NF+FF profiles
- Depolarization ratio with bias correction
- Raman effective attenuation slope
- Uncertainty propagation (GUM form with covariance preservation)
- Valid-data QC mask (time-range categorical)

### Flask Integration (app_v2.py)

Thin client over processing layer:
- `/api/process`: Accept L0 file → return L1/L2/L3 products as JSON
- `/api/config`: Return instrument configuration
- `/api/export/<level>/<channel>`: Export product to CSV

Same processing code used for:
- **Exploratory Flask UI** (algorithm validation)
- **Semi-production CLI** (batch processing)

## Usage

### As a Library

```python
from pathlib import Path
from v2.config import load_config
from v2.io import read_l0_from_csv
from v2.processing import L0bMatcher, L1Processor, L2Processor, L3Processor

# Load configuration
config = load_config(Path("v2/config/PALS_SBS312.json"))

# Read L0
l0 = read_l0_from_csv(Path("cruisedata/D2_1/D2_1 (D)_1.csv"))

# Produce L0b (PM enrichment)
matcher = L0bMatcher(pm_data_dir=Path("cruisedata/D2_1"))
l0b = matcher.process(l0)

# L1 (engineering corrected)
l1_proc = L1Processor(config)
l1 = l1_proc.process(l0b)

# L2 (calibrated profiles)
l2_proc = L2Processor(config)
l2 = l2_proc.process([l1])  # Ensemble of L1 shots

# L3 (final products)
l3_proc = L3Processor(config)
l3 = l3_proc.process(l2)

# Export to CSV
l3.to_dataframe("NF_CO").to_csv("output/L3_NF_CO.csv")
```

### Via Flask

```bash
cd v2
python app_v2.py
```

Then POST to `http://localhost:5000/api/process`:
```json
{
  "l0_file": "cruisedata/D2_1/D2_1 (D)_1.csv",
  "pm_dir": "cruisedata/D2_1",
  "levels": ["l1", "l2", "l3"]
}
```

### Via CLI (TBD)

```bash
python -m v2.cli process \
  --input cruisedata/D2_1/*.csv \
  --output results/ \
  --levels l1 l2 l3 \
  --config v2/config/PALS_SBS312.json
```

## Development Plan

**Phase 1 (Current)**: Set up directory structure, types, calibration classes, config schema
**Phase 2**: Implement L0 ingester (`pals_io.py` improvements for real CSV formats)
**Phase 3**: Integrate L0b matcher (around `match_power_data.py`)
**Phase 4**: Flesh out L1 processor (trigger reg, background, energy/gain norm with real calibrations)
**Phase 5**: Flesh out L2 processor (overlap, linearity, diagnostics)
**Phase 6**: Flesh out L3 processor (NF/FF stitching, depol, Raman, uncertainty combination)
**Phase 7**: Flask integration and testing
**Phase 8**: CLI wrapper
**Phase 9**: Documentation and deployment

## Uncertainty Propagation

**Principle**: Track component-wise uncertainties through each level.

**L1**: photon + background + energy + gain
**L2**: ensemble variance + calibration propagation
**L3**: Full GUM form with covariance preservation (NF/FF stitching, shared calibration)

**Output Format** (UNC-ready for NPL/CoMet):
- Value, standard uncertainty, distribution type, correlation class, dependencies, QC mask

## Integration with Existing Code

- **pmt.py**, **r9880U.py**: Imported by `calibration/classes.py` for responsivity/gain lookups
- **match_power_data.py**: Wrapped by `L0bMatcher` (separate module, not modified)
- **pypals.py**: Existing Flask app; v2 is parallel development
- **debug_matching.py**, **debug_loading.py**: Still available for validation

## Next Steps

1. **Review structure** - Does this architecture meet requirements?
2. **Phase 1 refinements** - Any changes to types, config schema, calibration classes?
3. **Begin Phase 2** - Implement L0 ingester based on actual CSV formats in `cruisedata/`

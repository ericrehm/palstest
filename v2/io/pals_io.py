"""
I/O module for reading and writing PALS data at each processing level.
Handles CSV, HDF5, and other formats.
"""

import csv
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, TextIO, Tuple
import numpy as np
import pandas as pd
from ..data_types import L0Data, L0bData, L1Data


# Real shot files interleave all six channels by row, identified by the
# Identity column. Identity strings ARE the channel_id used everywhere
# (config keys, calibration lookups, output columns) — no translation table.
KNOWN_IDENTITIES = {'co_near', 'cross_near', 'raman_near', 'co_far', 'cross_far', 'raman_far'}

_PMT_GAIN_HEADER_RE = re.compile(r'^#\s*pmt_gain_(\d+)\s*:\s*(-?[\d.]+)')


def _parse_pmt_gain_header(metadata_lines: List[str]) -> Dict[int, float]:
    """Parse '# pmt_gain_N: <value>' header lines into {N: value_volts}.

    These are the actual per-run PMT HV/gain setpoints the instrument
    recorded for this file -- operational data, not calibration data, so
    they live in the file header rather than PALS_SBS312.json. Each
    channel's config.pmt_gain_index says which N is its own voltage.
    """
    pmt_gains: Dict[int, float] = {}
    for line in metadata_lines:
        m = _PMT_GAIN_HEADER_RE.match(line)
        if m:
            pmt_gains[int(m.group(1))] = float(m.group(2))
    return pmt_gains


def parse_l0b_shots(lines: Iterable[str], source_file: str = "") -> List[L0bData]:
    """
    Parse an L0 or L0b shot CSV into one L0bData per row. L0 and L0b share the
    same on-disk format (L0b just has one extra '# L0b processed:' comment
    line and updated PowerASum/PowerBSum), so this reads either.

    Each row is a single channel's shot for one Timestamp, identified by the
    Identity column (see KNOWN_IDENTITIES) — rows are not correlated across
    Timestamps; each is processed independently.
    """
    lines_iter = iter(lines)
    metadata_lines: List[str] = []
    header_line = None
    for line in lines_iter:
        stripped = line.rstrip('\n')
        if stripped.startswith('#'):
            metadata_lines.append(stripped)
        else:
            header_line = stripped
            break

    if not header_line:
        raise ValueError(f"No header found in {source_file or '<stream>'}")

    header = [c.strip() for c in header_line.split(',')]
    bin_cols = sorted(
        (c for c in header if c.startswith('Bin_')),
        key=lambda c: int(c.split('_')[1]),
    )
    pmt_gain_by_index = _parse_pmt_gain_header(metadata_lines)  # file-level, parsed once

    shots: List[L0bData] = []
    reader = csv.DictReader(lines_iter, fieldnames=header)
    for row in reader:
        if not row or not row.get('Timestamp'):
            continue

        identity = (row.get('Identity') or '').strip()
        if identity not in KNOWN_IDENTITIES:
            continue  # unrecognized identity; skip rather than guess a channel
        channel_id = identity

        try:
            timestamp_ms = int(float(row['Timestamp']))
            adc_counts = {
                channel_id: np.array([float(row[c]) for c in bin_cols], dtype=float)
            }
        except (ValueError, TypeError, KeyError):
            continue

        timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

        l0 = L0Data(
            timestamp_utc=timestamp_utc,
            sequence_id=timestamp_ms,
            adc_counts=adc_counts,
            metadata={
                'source_file': source_file,
                'header_lines': metadata_lines,
                'pmt_gain_by_index': pmt_gain_by_index,
                'identity': identity,
                'roll': float(row.get('Roll') or 0),
                'pitch': float(row.get('Pitch') or 0),
                'yaw': float(row.get('Yaw') or 0),
                'laser_temp_c': float(row.get('LaserTempC') or 0),
            },
        )

        shots.append(L0bData(
            l0=l0,
            power_a_sum=float(row.get('PowerASum') or 0),
            power_b_sum=float(row.get('PowerBSum') or 0),
            pm_source='from_l0b_file',
        ))

    return shots


def read_l0b_shots_from_csv(csv_file: Path) -> List[L0bData]:
    """Read an L0 or L0b shot CSV from disk. See parse_l0b_shots()."""
    with open(csv_file, 'r') as f:
        return parse_l0b_shots(f, source_file=str(csv_file))


def read_l0_from_csv(csv_file: Path) -> L0Data:
    """
    Read L0 data from a CSV shot file.
    
    Expected CSV structure (from existing cruisedata/ files):
    - Columns: timestamp, sample_0, sample_1, ..., sample_1999 (for one channel)
    - OR: Columns for multiple channels (NF_CO, NF_CROSS, NF_RAMAN, FF_CO, FF_CROSS, FF_RAMAN)
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        L0Data object with parsed ADC counts and metadata
    """
    df = pd.read_csv(csv_file, comment='#')
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        timestamp_str = df['timestamp'].iloc[0]
    else:
        # Fallback: use file modification time
        timestamp_str = None
    
    try:
        if timestamp_str:
            timestamp_utc = datetime.fromisoformat(timestamp_str)
        else:
            timestamp_utc = datetime.fromtimestamp(csv_file.stat().st_mtime)
    except:
        timestamp_utc = datetime.now()
    
    # Infer channel structure from column names
    # Assume columns are: 'timestamp', 'channel_0', 'channel_1', etc.
    # or channel names like 'NF_CO_sample_0', 'NF_CO_sample_1', etc.
    
    adc_counts = {}
    
    # Look for sample columns (generic: sample_0, sample_1, ...)
    sample_cols = [c for c in df.columns if c.startswith('sample_')]
    if sample_cols:
        # Single-channel file
        channel_id = csv_file.stem  # Use filename as channel ID
        sample_indices = sorted([int(c.split('_')[1]) for c in sample_cols])
        adc_counts[channel_id] = df[[f'sample_{i}' for i in sample_indices]].iloc[0].values.astype(np.int16)
    else:
        # Multi-channel or channel-specific naming
        # Try to identify channel patterns
        channels = ['NF_CO', 'NF_CROSS', 'NF_RAMAN', 'FF_CO', 'FF_CROSS', 'FF_RAMAN']
        for ch in channels:
            ch_cols = [c for c in df.columns if c.startswith(ch)]
            if ch_cols:
                sample_indices = sorted([int(c.split('_')[-1]) for c in ch_cols if c.split('_')[-1].isdigit()])
                adc_counts[ch] = df[[f'{ch}_{i}' for i in sample_indices]].iloc[0].values.astype(np.int16)
    
    # If still empty, assume entire numeric data is waveform
    if not adc_counts:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        adc_counts[csv_file.stem] = df[numeric_cols].iloc[0].values.astype(np.int16)
    
    # Extract metadata if available
    metadata = {
        'source_file': str(csv_file),
        'rows': len(df),
    }
    if 'pulse_energy' in df.columns:
        metadata['pulse_energy_monitor_raw'] = float(df['pulse_energy'].iloc[0])
    
    return L0Data(
        timestamp_utc=timestamp_utc,
        sequence_id=0,  # TBD: extract from metadata
        adc_counts=adc_counts,
        metadata=metadata,
    )


def read_l0_batch(directory: Path, channel_filter: Optional[List[str]] = None) -> List[L0Data]:
    """
    Read multiple L0 files from a directory.
    
    Args:
        directory: Path to directory containing CSV files
        channel_filter: If provided, only read files matching these channel names
        
    Returns:
        List of L0Data objects
    """
    csv_files = sorted(directory.glob('*.csv'))
    
    l0_list = []
    for csv_file in csv_files:
        try:
            l0 = read_l0_from_csv(csv_file)
            if channel_filter is None or any(ch in str(csv_file) for ch in channel_filter):
                l0_list.append(l0)
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")
    
    return l0_list


def write_l0_to_csv(l0: L0Data, output_path: Path):
    """Write L0Data back to CSV for validation/inspection"""
    data_dict = {}
    
    # Flatten ADC counts into columns
    for channel_id, adc_array in l0.adc_counts.items():
        for i, count in enumerate(adc_array):
            data_dict[f'{channel_id}_sample_{i}'] = count
    
    df = pd.DataFrame([data_dict])
    df.insert(0, 'timestamp_utc', l0.timestamp_utc.isoformat())
    df.insert(1, 'sequence_id', l0.sequence_id)
    
    df.to_csv(output_path, index=False)
    print(f"L0 written to {output_path}")


def write_l0b_to_csv(l0b, output_path: Path):
    """
    Write L0bData to CSV preserving original shot file structure.
    Updates PowerASum/PowerBSum with matched values.
    Adds L0b provenance metadata as comments.
    """
    # Read original CSV to preserve all columns and structure
    original_file = Path(__file__).parent.parent.parent / "testdata" / "D3_1(D) transit_3.csv"
    
    # Try to find the original file from the l0b metadata
    if hasattr(l0b, 'l0') and hasattr(l0b.l0, 'metadata'):
        original_path = l0b.l0.metadata.get('source_file')
        if original_path and Path(original_path).exists():
            original_file = Path(original_path)
    
    # Read original file preserving structure
    metadata_lines = []
    header_line = None
    data_lines = []
    
    try:
        with open(original_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    metadata_lines.append(line.rstrip())
                elif not header_line:
                    header_line = line.rstrip()
                else:
                    data_lines.append(line.rstrip())
    except Exception as e:
        print(f"Warning: Could not read original file {original_file}: {e}")
        # Fallback: create minimal CSV
        metadata_lines = []
        header_line = "Timestamp,PowerASum,PowerBSum"
        data_lines = []
    
    # Write L0b file with metadata and updated power values
    with open(output_path, 'w') as f:
        # Write L0b provenance metadata
        f.write(f"# L0b Processing Output\n")
        f.write(f"# Source Timestamp: {l0b.l0.timestamp_utc.isoformat()}\n")
        f.write(f"# Sequence ID: {l0b.l0.sequence_id}\n")
        f.write(f"# PM Source: {l0b.pm_source}\n")
        f.write(f"# Match Quality: {l0b.match_quality}\n")
        f.write(f"# Power A Sum (Matched): {l0b.power_a_sum}\n")
        f.write(f"# Power B Sum (Matched): {l0b.power_b_sum}\n")
        
        # Write original metadata
        for line in metadata_lines:
            f.write(line + "\n")
        
        # Write header
        if header_line:
            f.write(header_line + "\n")
        
        # Write data with updated PowerASum/PowerBSum
        if header_line:
            headers = [col.strip() for col in header_line.split(',')]
            power_a_idx = headers.index('PowerASum') if 'PowerASum' in headers else -1
            power_b_idx = headers.index('PowerBSum') if 'PowerBSum' in headers else -1
            
            for data_line in data_lines:
                values = data_line.split(',')
                
                # Update power values if columns exist
                if power_a_idx >= 0 and power_a_idx < len(values):
                    values[power_a_idx] = str(l0b.power_a_sum)
                if power_b_idx >= 0 and power_b_idx < len(values):
                    values[power_b_idx] = str(l0b.power_b_sum)
                
                f.write(','.join(values) + "\n")
    
    print(f"L0b written to {output_path}")


# QA test name -> (value column prefix, flag column prefix). A test with no
# scalar value (flag-only) can omit the first; see L1Data.qa_values/qa_flags.
# Add an entry here for each new QA test as it's implemented.
QA_COLUMN_PREFIXES = {
    "adc_occupancy": ("QADC", "QADCFLAG"),
}


def write_l1_shots(l1_shots: List[L1Data], out: TextIO, stage_label: str = "L1") -> None:
    """
    Write shots in the same row-per-shot format as L0/L0b ("the Matrix"):
    original metadata comment lines propagated forward, one new
    '# <stage_label> processed:' line, Bin_ values from shot.signal, Range_
    columns appended (eqn 1), and one pair of columns per QA test that ran
    (e.g. QADC_/QADCFLAG_ for eqn 9's ADC occupancy -- see
    QA_COLUMN_PREFIXES and L1Data.qa_values/qa_flags).

    Each L1Data here wraps a single-channel shot (one CSV row in, one out),
    matching the real file format where Identity distinguishes channels.

    stage_label only changes the provenance comment line -- this function
    also writes L2's per-shot range-corrected output (see app_v2.run_l2),
    which reuses L1Data with shot.signal swapped for eqn 7's X_j(r) =
    r^2 * S_hat_j(r), plus one extra ensemble-SNR synthetic-Identity row per
    channel (see L2Processor.build_ensemble_snr).
    """
    if not l1_shots:
        raise ValueError("No L1 shots to write")

    header_lines = l1_shots[0].l0b.l0.metadata.get('header_lines', [])
    first_channel = next(iter(l1_shots[0].signal))
    n_bins = len(l1_shots[0].signal[first_channel])
    bin_cols = [f'Bin_{i}' for i in range(n_bins)]
    range_cols = [f'Range_{i}' for i in range(n_bins)]

    # S-tilde (eqn 5, pre-gain-normalization) is otherwise lost once written
    # to disk -- Bin_ only ever holds one signal representation (S-hat for
    # L1, X_j for L2). Anything that needs to re-derive S-tilde from a
    # written file (parse_l1_shots, and anything built on it -- e.g.
    # re-running L2Processor.build_ensemble_snr against an uploaded L1 file
    # in the split-per-stage-request architecture) would otherwise silently
    # get S-hat/X_j instead, quietly reintroducing the gain-model dependence
    # S-tilde was chosen to avoid. Scan every shot, not just the first --
    # same reasoning as the QA columns below.
    stilde_present = any(shot.signal_tilde for shot in l1_shots)
    stilde_cols = [f'STILDE_{i}' for i in range(n_bins)] if stilde_present else []

    # One column block per QA test present anywhere in the batch, in whatever
    # value/flag shape it has. Not every shot has every test -- e.g.
    # L1CrossProcessor's derived rows (qer_near, depol_near, ...) only carry
    # a borrowed adc_occupancy *flag*, no value, and no other test at all --
    # so this must scan every shot, not just the first.
    values_present = set()
    flags_present = set()
    for shot in l1_shots:
        values_present.update(shot.qa_values)
        flags_present.update(shot.qa_flags)
    qa_test_names = sorted(values_present | flags_present)

    qa_value_cols: dict[str, list[str]] = {}
    qa_flag_cols: dict[str, list[str]] = {}
    qa_cols: list[str] = []
    for test_name in qa_test_names:
        value_prefix, flag_prefix = QA_COLUMN_PREFIXES.get(
            test_name, (test_name.upper(), f'{test_name.upper()}_FLAG')
        )
        if test_name in values_present:
            cols = [f'{value_prefix}_{i}' for i in range(n_bins)]
            qa_value_cols[test_name] = cols
            qa_cols += cols
        if test_name in flags_present:
            cols = [f'{flag_prefix}_{i}' for i in range(n_bins)]
            qa_flag_cols[test_name] = cols
            qa_cols += cols

    fieldnames = [
        'Timestamp', 'Identity', 'Roll', 'Pitch', 'Yaw', 'LaserTempC',
        'PowerASum', 'PowerBSum',
    ] + bin_cols + range_cols + stilde_cols + qa_cols

    for line in header_lines:
        out.write(line + '\n')
    out.write(f"# {stage_label} processed: {datetime.now(timezone.utc).isoformat()}\n")
    out.write(','.join(fieldnames) + '\n')

    writer = csv.DictWriter(out, fieldnames=fieldnames)
    for shot in l1_shots:
        l0 = shot.l0b.l0
        channel_id = next(iter(shot.signal))
        row = {
            'Timestamp': l0.sequence_id,
            'Identity': l0.metadata.get('identity', ''),
            'Roll': l0.metadata.get('roll', ''),
            'Pitch': l0.metadata.get('pitch', ''),
            'Yaw': l0.metadata.get('yaw', ''),
            'LaserTempC': l0.metadata.get('laser_temp_c', ''),
            'PowerASum': shot.l0b.power_a_sum,
            'PowerBSum': shot.l0b.power_b_sum,
        }
        row.update(zip(bin_cols, shot.signal[channel_id]))
        row.update(zip(range_cols, shot.range_m[channel_id]))
        if stilde_cols:
            s_tilde = shot.signal_tilde.get(channel_id)
            if s_tilde is not None:
                row.update(zip(stilde_cols, s_tilde))
        # Not every shot has every QA test (see the comment above) -- leave a
        # row's columns for a missing test unset; DictWriter fills blanks.
        for test_name, cols in qa_value_cols.items():
            values = shot.qa_values.get(test_name, {}).get(channel_id)
            if values is not None:
                row.update(zip(cols, values))
        for test_name, cols in qa_flag_cols.items():
            values = shot.qa_flags.get(test_name, {}).get(channel_id)
            if values is not None:
                row.update(zip(cols, values))
        writer.writerow(row)


def write_l1_shots_to_csv(l1_shots: List[L1Data], output_path: Path) -> None:
    """Write L1 shots to a CSV file on disk. See write_l1_shots()."""
    with open(output_path, 'w', newline='') as f:
        write_l1_shots(l1_shots, f)
    print(f"L1 written to {output_path} ({len(l1_shots)} shots)")


_INDEXED_COLUMN_RE = re.compile(r'^(.+)_(\d+)$')


def _group_indexed_columns(header: List[str]) -> Dict[str, List[str]]:
    """Group column names like 'Bin_0'..'Bin_399' by prefix, ordered by index."""
    indexed: Dict[str, List[Tuple[int, str]]] = {}
    for col in header:
        m = _INDEXED_COLUMN_RE.match(col)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        indexed.setdefault(prefix, []).append((idx, col))
    return {prefix: [c for _, c in sorted(cols)] for prefix, cols in indexed.items()}


def parse_l1_shots(lines: Iterable[str], source_file: str = "") -> List[L1Data]:
    """
    Parse an L1 or L2 shot CSV (as written by write_l1_shots) back into
    L1Data objects -- the inverse of write_l1_shots. Needed so a later stage
    can be run against an uploaded L1/L2 file directly: e.g. L2 alone from
    an existing L1 file, or splitting L0b->L1->L2 into separate HTTP
    requests (each returning one stage's CSV as a raw response body instead
    of bundling all of them into one JSON blob that can reach 1+ GB for
    large files -- see app_v2.py).

    L1 and L2 files share the same column layout (L2's Bin_ holds eqn 7's
    X_j instead of S-hat, but the schema is identical), so this reads
    either -- the result's .signal holds whatever was in Bin_, whichever
    that was. .signal_tilde is only populated if an STILDE_ column block is
    present (write_l1_shots only writes one when signal_tilde was actually
    set -- see its comment); code that needs S-tilde specifically (e.g.
    L2Processor.build_ensemble_snr) will KeyError rather than silently
    substitute S-hat/X_j if it's missing.

    QA columns (QADC_/QADCFLAG_/DETECTOR_SNR_/...) are recovered generically:
    any other indexed column block is matched against QA_COLUMN_PREFIXES'
    registered prefixes first, falling back to inferring a value-only test
    from the column prefix (or a flag-only test if the prefix ends in
    "_FLAG") -- so a new QA test written with the default naming convention
    is read back automatically, with no change needed here.
    """
    lines_iter = iter(lines)
    metadata_lines: List[str] = []
    header_line = None
    for line in lines_iter:
        stripped = line.rstrip('\n')
        if stripped.startswith('#'):
            metadata_lines.append(stripped)
        else:
            header_line = stripped
            break

    if not header_line:
        raise ValueError(f"No header found in {source_file or '<stream>'}")

    header = [c.strip() for c in header_line.split(',')]
    groups = _group_indexed_columns(header)

    bin_cols = groups.get('Bin')
    range_cols = groups.get('Range')
    if not bin_cols or not range_cols:
        raise ValueError(f"Expected Bin_/Range_ columns in {source_file or '<stream>'}")
    stilde_cols = groups.get('STILDE')

    # Inverse of QA_COLUMN_PREFIXES, plus a generic fallback (see docstring)
    # for any other indexed column block -- e.g. DETECTOR_SNR_, which uses
    # the fallback naming convention rather than an explicit registration.
    prefix_to_test: Dict[str, Tuple[str, str]] = {}
    for test_name, (value_prefix, flag_prefix) in QA_COLUMN_PREFIXES.items():
        prefix_to_test[value_prefix] = ('value', test_name)
        prefix_to_test[flag_prefix] = ('flag', test_name)

    qa_groups: Dict[str, Tuple[str, str, List[str]]] = {}
    for prefix, cols in groups.items():
        if prefix in ('Bin', 'Range', 'STILDE'):
            continue
        if prefix in prefix_to_test:
            kind, test_name = prefix_to_test[prefix]
        elif prefix.endswith('_FLAG'):
            kind, test_name = 'flag', prefix[:-len('_FLAG')].lower()
        else:
            kind, test_name = 'value', prefix.lower()
        qa_groups[prefix] = (kind, test_name, cols)

    shots: List[L1Data] = []
    reader = csv.DictReader(lines_iter, fieldnames=header)
    for row in reader:
        if not row or not row.get('Timestamp'):
            continue

        try:
            timestamp_ms = int(float(row['Timestamp']))
            signal_arr = np.array([float(row[c]) for c in bin_cols], dtype=float)
            range_arr = np.array([float(row[c]) for c in range_cols], dtype=float)
        except (ValueError, TypeError, KeyError):
            continue

        channel_id = (row.get('Identity') or '').strip()

        signal_tilde: Dict[str, np.ndarray] = {}
        if stilde_cols and all(row.get(c) not in (None, '') for c in stilde_cols):
            try:
                signal_tilde[channel_id] = np.array([float(row[c]) for c in stilde_cols], dtype=float)
            except ValueError:
                pass

        qa_values: Dict[str, Dict[str, np.ndarray]] = {}
        qa_flags: Dict[str, Dict[str, np.ndarray]] = {}
        for kind, test_name, cols in qa_groups.values():
            raw_vals = [v for v in (row.get(c) for c in cols) if v not in (None, '')]
            if len(raw_vals) != len(cols):
                continue  # test wasn't populated for this row (e.g. QADC_ on a derived-identity row)
            try:
                arr = np.array([float(v) for v in raw_vals], dtype=float)
            except ValueError:
                continue
            (qa_flags if kind == 'flag' else qa_values).setdefault(test_name, {})[channel_id] = arr

        timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        l0 = L0Data(
            timestamp_utc=timestamp_utc,
            sequence_id=timestamp_ms,
            adc_counts={},  # raw counts aren't present in an L1/L2 file
            metadata={
                'source_file': source_file,
                'header_lines': metadata_lines,
                'identity': channel_id,
                'roll': float(row.get('Roll') or 0),
                'pitch': float(row.get('Pitch') or 0),
                'yaw': float(row.get('Yaw') or 0),
                'laser_temp_c': float(row.get('LaserTempC') or 0),
            },
        )
        l0b = L0bData(
            l0=l0,
            power_a_sum=float(row.get('PowerASum') or 0),
            power_b_sum=float(row.get('PowerBSum') or 0),
            pm_source='from_l1_file',
        )
        shots.append(L1Data(
            l0b=l0b,
            signal={channel_id: signal_arr},
            signal_tilde=signal_tilde,
            range_m={channel_id: range_arr},
            qa_values=qa_values,
            qa_flags=qa_flags,
        ))

    return shots


def read_l1_shots_from_csv(csv_file: Path) -> List[L1Data]:
    """Read an L1 or L2 shot CSV from disk. See parse_l1_shots()."""
    with open(csv_file, 'r') as f:
        return parse_l1_shots(f, source_file=str(csv_file))


def write_l1_to_csv(l1, channel_id: str, output_path: Path):
    """Write L1Data channel to CSV (via to_dataframe)"""
    df = l1.to_dataframe(channel_id, include_uncertainty=True)
    df.to_csv(output_path, index=False)
    print(f"L1 {channel_id} written to {output_path}")


def write_l2_to_csv(l2, channel_id: str, output_path: Path):
    """Write L2Data channel to CSV (via to_dataframe)"""
    df = l2.to_dataframe(channel_id)
    df.to_csv(output_path, index=False)
    print(f"L2 {channel_id} written to {output_path}")


def write_l3_to_csv(l3, channel_id: str, output_path: Path, include_unc_breakdown: bool = True):
    """Write L3Data channel to CSV in UNC-compatible format"""
    df = l3.to_dataframe(channel_id, include_unc_breakdown=include_unc_breakdown)
    df.to_csv(output_path, index=False)
    print(f"L3 {channel_id} written to {output_path} (UNC-ready)")


# HDF5 support (optional, for efficient storage)
try:
    import h5py
    
    def write_l1_to_hdf5(l1, output_path: Path):
        """Write L1Data to HDF5 for efficient storage"""
        with h5py.File(output_path, 'w') as f:
            # Metadata group
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['timestamp_utc'] = l1.l0b.l0.timestamp_utc.isoformat()
            meta_grp.attrs['sequence_id'] = l1.l0b.l0.sequence_id
            
            # Signal group
            sig_grp = f.create_group('signal')
            for ch_id, sig in l1.signal.items():
                sig_grp.create_dataset(ch_id, data=sig)
            
            # Uncertainty group
            unc_grp = f.create_group('uncertainty')
            for ch_id, unc in l1.uncertainty_total.items():
                unc_grp.create_dataset(ch_id, data=unc)
            
            # Flags group
            flag_grp = f.create_group('flags')
            for ch_id, flags in l1.saturation_flags.items():
                flag_grp.create_dataset(ch_id, data=flags)
        
        print(f"L1 written to HDF5: {output_path}")

except ImportError:
    def write_l1_to_hdf5(*args, **kwargs):
        raise ImportError("h5py not installed; HDF5 support unavailable")

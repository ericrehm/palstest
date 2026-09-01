"""
Peak Analysis vs Power
Command-line tool: for the three natural far-field identities (co_far,
cross_far, raman_far), finds each shot's early-bin waveform peak and plots it
against that shot's PM-corrected power (PowerASum/PowerBSum).

Works on any pipeline level -- raw, L0b, L1, or L2 -- since they all share
the same on-disk "Matrix" CSV shape (Timestamp/Identity/PowerASum/PowerBSum/
Bin_N...; see v2/io/pals_io.py's parse_l0b_shots). "Bin_N" means raw ADC
counts for raw/L0b, and gain-normalized/range-corrected signal (S-hat/X_j)
for L1/L2 -- write_l1_shots carries PowerASum/PowerBSum forward unchanged
from the original raw file at every level, so the same peak-vs-power
question is meaningful throughout, just against a different Y quantity.
"""
import argparse
import re
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from v2.io import parse_l0b_shots

FAR_FIELD_IDENTITIES = ['co_far', 'cross_far', 'raman_far']
COLORS = {'co_far': '#0ea5e9', 'cross_far': '#ef4444', 'raman_far': '#8b5cf6'}
LEVELS = ['raw', 'l0b', 'l1', 'l2']

# Same PM-sibling-file convention used elsewhere (index_v2.html, viewer.html)
# -- a directory scan should skip these rather than trying to analyze them
# as shot files (they have a totally different column schema).
_PM_FILE_RE = re.compile(r'PM|power.?monitor|_power\.csv$', re.IGNORECASE)

# Same output-filename tagging convention used by app_v2.py/index_v2.html.
_LEVEL_TAG_RE = {
    'l0b': re.compile(r'_L0b_\d+\.csv$', re.IGNORECASE),
    'l1': re.compile(r'_L1_\d+\.csv$', re.IGNORECASE),
    'l2': re.compile(r'_L2_\d+\.csv$', re.IGNORECASE),
}

# Only for labeling (title/axis text) -- "Bin_N" is raw ADC counts at
# raw/L0b, but already gain-normalized/range-corrected signal by L1/L2.
_VALUE_LABEL = {
    'raw': 'Peak Value (ADC counts)',
    'l0b': 'Peak Value (ADC counts)',
    'l1': 'Peak Value (Signal)',
    'l2': 'Peak Value (Signal)',
}


def detect_level(filename: str) -> str:
    """Level implied by the standard output-filename tag; untagged means raw."""
    for level, pattern in _LEVEL_TAG_RE.items():
        if pattern.search(filename):
            return level
    return 'raw'


def analyze_file(path: Path, end_bin: int) -> Dict[str, List[dict]]:
    """
    Parse one L0b (or raw) shot file and, for each of the three far-field
    identities, find the peak (max value) within bins [0, end_bin) of every
    shot -- the "early peak" -- alongside that shot's PM-corrected power.
    """
    with open(path, 'r') as f:
        shots = parse_l0b_shots(f, source_file=str(path))

    results: Dict[str, List[dict]] = {identity: [] for identity in FAR_FIELD_IDENTITIES}
    for shot in shots:
        identity = shot.l0.metadata.get('identity', '')
        if identity not in FAR_FIELD_IDENTITIES:
            continue
        waveform = shot.l0.adc_counts.get(identity)
        if waveform is None or not len(waveform):
            continue

        window = waveform[:end_bin]
        if not len(window):
            continue
        peak_bin = int(np.argmax(window))
        peak_value = float(window[peak_bin])

        results[identity].append({
            'timestamp_ms': int(shot.l0.timestamp_utc.timestamp() * 1000),
            'power_a': shot.power_a_sum,
            'power_b': shot.power_b_sum,
            'peak_bin': peak_bin,
            'peak_value': peak_value,
        })

    return results


def print_peak_statistics(path: Path, results: Dict[str, List[dict]], end_bin: int, level: str) -> None:
    """Print summary statistics of peak analysis."""
    print(f"\n{'=' * 60}")
    print(f"PEAK ANALYSIS SUMMARY: {path.name} [{level.upper()}] (bins 0-{end_bin})")
    print(f"{'=' * 60}")

    for identity, data in results.items():
        if not data:
            print(f"\n{identity}: No data")
            continue

        peak_values = [d['peak_value'] for d in data]
        power_a_vals = [d['power_a'] for d in data]
        power_b_vals = [d['power_b'] for d in data]

        # :.4g (significant figures), not a fixed decimal count -- raw/L0b
        # peaks are ADC-count-scale (hundreds to thousands) but L1/L2 peaks
        # are gain-normalized signal, which can be ~1e-4 or smaller. A fixed
        # ".1f" rounds those to a misleading "0.0" instead of showing them.
        print(f"\n{identity}:")
        print(f"  Records: {len(data)}")
        print(f"  Peak Value: min={min(peak_values):.4g}, max={max(peak_values):.4g}, mean={sum(peak_values) / len(peak_values):.4g}")
        print(f"  PowerA: min={min(power_a_vals):.2f}, max={max(power_a_vals):.2f}, mean={sum(power_a_vals) / len(power_a_vals):.2f}")
        print(f"  PowerB: min={min(power_b_vals):.2f}, max={max(power_b_vals):.2f}, mean={sum(power_b_vals) / len(power_b_vals):.2f}")

    print(f"{'=' * 60}\n")


def build_figure(results: Dict[str, List[dict]], title: str, end_bin: int, level: str) -> go.Figure:
    """PowerASum and PowerBSum side by side in one HTML page, one trace per
    far-field identity in each, sharing a legend (toggle an identity once,
    it hides in both panels)."""
    value_label = _VALUE_LABEL[level]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Peak vs PowerASum (bins 0-{end_bin})', f'Peak vs PowerBSum (bins 0-{end_bin})'),
    )

    for identity, data in results.items():
        if not data:
            continue
        color = COLORS.get(identity, '#666')
        peak_vals = [d['peak_value'] for d in data]
        hover_text = [f"Bin {d['peak_bin']}, TS {d['timestamp_ms']}" for d in data]

        fig.add_trace(go.Scatter(
            x=[d['power_a'] for d in data], y=peak_vals,
            mode='markers', name=identity, legendgroup=identity,
            marker=dict(size=8, color=color), text=hover_text,
            hovertemplate=f'<b>%{{fullData.name}}</b><br>PowerA: %{{x:.2f}}<br>{value_label}: %{{y}}<br>%{{text}}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[d['power_b'] for d in data], y=peak_vals,
            mode='markers', name=identity, legendgroup=identity, showlegend=False,
            marker=dict(size=8, color=color), text=hover_text,
            hovertemplate=f'<b>%{{fullData.name}}</b><br>PowerB: %{{x:.2f}}<br>{value_label}: %{{y}}<br>%{{text}}<extra></extra>',
        ), row=1, col=2)

    fig.update_xaxes(title_text='PowerASum', row=1, col=1)
    fig.update_xaxes(title_text='PowerBSum', row=1, col=2)
    fig.update_yaxes(title_text=_VALUE_LABEL[level], row=1, col=1)
    fig.update_layout(
        title=f'Early-Peak vs Power — {title} ({level.upper()})',
        height=560, width=1400, hovermode='closest', template='plotly_dark',
    )
    return fig


def process_one(path: Path, end_bin: int, do_plot: bool, do_browser: bool, level: Optional[str]) -> None:
    resolved_level = level or detect_level(path.name)
    print(f"Loading {path} (level={resolved_level}) ...")
    results = analyze_file(path, end_bin)
    print_peak_statistics(path, results, end_bin, resolved_level)

    if not (do_plot or do_browser):
        return
    if not any(results.values()):
        print(f"  No far-field shots found in {path.name} -- skipping plot")
        return

    fig = build_figure(results, path.name, end_bin, resolved_level)

    if do_plot:
        out_path = path.with_name(f"{path.stem}_peaks_vs_power.html")
        fig.write_html(str(out_path))
        print(f"  Saved: {out_path}")
        if do_browser:
            webbrowser.open(out_path.resolve().as_uri())
    elif do_browser:
        # -b without -p: nothing was written to open, so use a scratch file.
        tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
        tmp.close()
        fig.write_html(tmp.name)
        webbrowser.open(Path(tmp.name).resolve().as_uri())


def discover_files(path: Path, level: Optional[str] = None) -> List[Path]:
    """
    A single file, or every non-PM .csv directly inside a directory (not
    recursive). `level`, if given, filters a directory scan to just that
    level's tagged files (untagged files count as 'raw'); for a single file
    it's not a filter at all -- process_one() uses it only to label the
    output, since the caller presumably knows what they picked.
    """
    if path.is_file():
        return [path]
    files = [p for p in path.glob('*.csv') if not _PM_FILE_RE.search(p.name)]
    if level is not None:
        files = [p for p in files if detect_level(p.name) == level]
    return sorted(files)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot early-bin peak waveform value vs PM power (PowerASum/PowerBSum) "
                    "for the co_far/cross_far/raman_far identities, for an L0b (or raw) shot "
                    "file or every such file in a directory."
    )
    parser.add_argument('path', type=Path, help='shot CSV file (any level), or a directory of them')
    parser.add_argument(
        '-e', '--end', type=int, default=50, dest='end_bin', metavar='BIN',
        help="End of the early-peak search range (bin index, exclusive). Default: 50",
    )
    parser.add_argument(
        '-l', '--level', choices=LEVELS, default=None,
        help="Pipeline level to analyze. In directory mode, restricts the scan to files "
             "tagged for this level (untagged files count as 'raw'); for a single file it "
             "only affects labeling (auto-detected from the filename otherwise). Default: "
             "all levels found (directory mode) / auto-detected (single file).",
    )
    parser.add_argument('-p', '--plot', action='store_true', help='Write an HTML plot file next to each input file')
    parser.add_argument('-b', '--browser', action='store_true', help='Open the plot in a browser')
    args = parser.parse_args(argv)

    if not args.path.exists():
        parser.error(f"Path not found: {args.path}")

    files = discover_files(args.path, args.level)
    if not files:
        parser.error(f"No CSV files found in {args.path}")

    for f in files:
        process_one(f, args.end_bin, args.plot, args.browser, args.level)

    return 0


if __name__ == '__main__':
    sys.exit(main())

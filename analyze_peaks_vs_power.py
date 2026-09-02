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


def _format_mmss(seconds: float) -> str:
    """Elapsed seconds as MM:SS, minutes uncapped (e.g. a run over an hour
    shows '75:03') -- not the same as clock-wall-time HH:MM:SS."""
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f'{minutes:02d}:{secs:02d}'


_TIME_TICK_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]


def _time_ticks(max_elapsed_s: float, target_ticks: int = 8):
    """Tick positions/labels for an elapsed-seconds axis rendered as mm:ss.
    Plotly's linear-axis tickformat is d3-format (numeric only, no minutes/
    seconds), so the mm:ss tick labels have to be built by hand rather than
    via a 'date' axis type (which would wrap minutes at each hour)."""
    step = _TIME_TICK_STEPS_S[-1]
    for candidate in _TIME_TICK_STEPS_S:
        if max_elapsed_s / candidate <= target_ticks:
            step = candidate
            break
    n_ticks = int(max_elapsed_s // step) + 2
    tickvals = [i * step for i in range(n_ticks)]
    ticktext = [_format_mmss(v) for v in tickvals]
    return tickvals, ticktext


def _lighten_hex(hex_color: str, amount: float = 0.45) -> str:
    """Same hue, shifted toward white by `amount` (0-1) -- used to give the
    Peak trace a shade of the identity color that still reads as "the same
    identity" but is visually distinct from the full-saturation PowerA/
    PowerB lines it shares the time-series plot with."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


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


def build_identity_figure(identity: str, data: List[dict], title: str, end_bin: int, level: str) -> go.Figure:
    """One identity's full layout: PowerASum/PowerBSum vs Peak side by side
    on top, and (spanning both columns below) PowerASum/PowerBSum vs elapsed
    time (mm:ss, left axis) with Peak vs the same time axis on the right --
    Peak is on a different scale per level (counts for L0b, Watts for L1/L2)
    while PowerASum/PowerBSum are always power-monitor arbitrary units.

    Kept to a single identity per figure (rather than overlaying all three)
    since three colors x three quantities in one legend was hard to read.
    """
    value_label = _VALUE_LABEL[level]
    color = COLORS.get(identity, '#666')
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{}, {}],
            [{'secondary_y': True, 'colspan': 2}, None],
        ],
        subplot_titles=(
            f'Peak vs PowerASum (bins 0-{end_bin})',
            f'Peak vs PowerBSum (bins 0-{end_bin})',
            'PowerASum (solid) / PowerBSum (dashed) / Peak (markers, right axis) vs Time',
        ),
        vertical_spacing=0.15,
    )

    peak_vals = [d['peak_value'] for d in data]
    hover_text = [f"Bin {d['peak_bin']}, TS {d['timestamp_ms']}" for d in data]

    fig.add_trace(go.Scatter(
        x=[d['power_a'] for d in data], y=peak_vals,
        mode='markers', name='Peak', showlegend=False,
        marker=dict(size=8, color=color), text=hover_text,
        hovertemplate=f'PowerA: %{{x:.2f}}<br>{value_label}: %{{y}}<br>%{{text}}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[d['power_b'] for d in data], y=peak_vals,
        mode='markers', name='Peak', showlegend=False,
        marker=dict(size=8, color=color), text=hover_text,
        hovertemplate=f'PowerB: %{{x:.2f}}<br>{value_label}: %{{y}}<br>%{{text}}<extra></extra>',
    ), row=1, col=2)

    t0_ms = min(d['timestamp_ms'] for d in data) if data else 0
    max_elapsed_s = (max(d['timestamp_ms'] for d in data) - t0_ms) / 1000.0 if data else 0.0
    elapsed_s = [(d['timestamp_ms'] - t0_ms) / 1000.0 for d in data]
    mmss_text = [_format_mmss(s) for s in elapsed_s]

    fig.add_trace(go.Scatter(
        x=elapsed_s, y=[d['power_a'] for d in data],
        mode='lines+markers', name='PowerA',
        line=dict(color=color, dash='solid'), marker=dict(size=4, color=color),
        text=mmss_text,
        hovertemplate='<b>PowerA</b><br>t=%{text}<br>PowerA: %{y:.2f}<extra></extra>',
    ), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=elapsed_s, y=[d['power_b'] for d in data],
        mode='lines+markers', name='PowerB',
        line=dict(color=color, dash='dash'), marker=dict(size=4, color=color, symbol='square'),
        text=mmss_text,
        hovertemplate='<b>PowerB</b><br>t=%{text}<br>PowerB: %{y:.2f}<extra></extra>',
    ), row=2, col=1, secondary_y=False)

    peak_color = _lighten_hex(color)
    fig.add_trace(go.Scatter(
        x=elapsed_s, y=peak_vals,
        mode='lines+markers', name='Peak',
        line=dict(color=peak_color, dash='dot'),
        marker=dict(size=7, color=peak_color, symbol='diamond', line=dict(width=1, color='white')),
        text=mmss_text,
        hovertemplate=f'<b>Peak</b><br>t=%{{text}}<br>{value_label}: %{{y}}<extra></extra>',
    ), row=2, col=1, secondary_y=True)

    tickvals, ticktext = _time_ticks(max_elapsed_s)
    fig.update_xaxes(
        title_text='Time (mm:ss)', tickmode='array', tickvals=tickvals, ticktext=ticktext,
        row=2, col=1,
    )
    fig.update_yaxes(title_text='PowerASum / PowerBSum', row=2, col=1, secondary_y=False)
    # Fixed +/-25%-of-mean range (not autoscaled) so the Peak axis's relative
    # spread is directly comparable across levels -- e.g. L0b (pre pulse-
    # energy normalization) vs L1 (post) -- even though their absolute scales
    # differ enormously (ADC counts vs Watts).
    mean_peak = sum(peak_vals) / len(peak_vals) if peak_vals else 0.0
    peak_range = sorted([mean_peak * 0.75, mean_peak * 1.25]) if mean_peak else None
    fig.update_yaxes(title_text=value_label, range=peak_range, row=2, col=1, secondary_y=True)

    fig.update_xaxes(title_text='PowerASum', row=1, col=1)
    fig.update_xaxes(title_text='PowerBSum', row=1, col=2)
    fig.update_yaxes(title_text=value_label, row=1, col=1)
    fig.update_layout(
        title=f'{identity} — Early-Peak vs Power — {title} ({level.upper()})',
        height=820, width=1400, hovermode='closest', template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.0),
    )
    return fig


def build_page_html(results: Dict[str, List[dict]], title: str, end_bin: int, level: str) -> str:
    """One HTML page holding a stack of independent per-identity figures
    (co_far, cross_far, raman_far), each with the full layout above."""
    sections = []
    for identity in FAR_FIELD_IDENTITIES:
        data = results.get(identity, [])
        if not data:
            continue
        fig = build_identity_figure(identity, data, title, end_bin, level)
        include_js = 'cdn' if not sections else False
        div_html = fig.to_html(full_html=False, include_plotlyjs=include_js)
        sections.append(div_html)
    body = '\n<hr style="border-color:#333; margin:32px 0;">\n'.join(sections)
    return (
        f'<html><head><meta charset="utf-8"><title>{title} — Peak vs Power ({level.upper()})</title></head>'
        f'<body style="background:#111111; margin:0;">{body}</body></html>'
    )


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

    html = build_page_html(results, path.name, end_bin, resolved_level)

    if do_plot:
        out_path = path.with_name(f"{path.stem}_peaks_vs_power.html")
        out_path.write_text(html)
        print(f"  Saved: {out_path}")
        if do_browser:
            webbrowser.open(out_path.resolve().as_uri())
    elif do_browser:
        # -b without -p: nothing was written to open, so use a scratch file.
        tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
        tmp.write(html)
        tmp.close()
        webbrowser.open(Path(tmp.name).resolve().as_uri())


def process_combined(paths: List[Path], end_bin: int, do_plot: bool, do_browser: bool, level: str, dir_path: Path) -> None:
    """Like process_one, but merges every file's shots into one combined
    dataset per identity (sorted back into timestamp order, since files are
    concatenated) and writes a single plot for the whole directory."""
    print(f"Loading {len(paths)} files from {dir_path} (level={level}) ...")
    for p in paths:
        print(f"  - {p.name}")

    results: Dict[str, List[dict]] = {identity: [] for identity in FAR_FIELD_IDENTITIES}
    for p in paths:
        file_results = analyze_file(p, end_bin)
        for identity, data in file_results.items():
            results[identity].extend(data)
    for data in results.values():
        data.sort(key=lambda d: d['timestamp_ms'])

    print_peak_statistics(dir_path, results, end_bin, level)

    if not (do_plot or do_browser):
        return
    if not any(results.values()):
        print(f"  No far-field shots found in {dir_path} -- skipping plot")
        return

    html = build_page_html(results, dir_path.name, end_bin, level)

    if do_plot:
        out_path = dir_path / f"{dir_path.name}_peaks_vs_power.html"
        out_path.write_text(html)
        print(f"  Saved: {out_path}")
        if do_browser:
            webbrowser.open(out_path.resolve().as_uri())
    elif do_browser:
        # -b without -p: nothing was written to open, so use a scratch file.
        tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
        tmp.write(html)
        tmp.close()
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
    parser.add_argument(
        '-a', '--all', action='store_true',
        help="Directory mode: combine every discovered file into one combined plot "
             "(shots merged and re-sorted by time) instead of writing one output per file.",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        parser.error(f"Path not found: {args.path}")

    files = discover_files(args.path, args.level)
    if not files:
        parser.error(f"No CSV files found in {args.path}")

    if args.all:
        if args.path.is_file():
            parser.error("-a/--all only applies to a directory of files")
        level = args.level
        if level is None:
            detected_levels = {detect_level(f.name) for f in files}
            if len(detected_levels) > 1:
                parser.error(
                    f"Files span multiple levels ({', '.join(sorted(detected_levels))}); "
                    "pass -l/--level to pick one for --all mode."
                )
            level = detected_levels.pop()
        process_combined(files, args.end_bin, args.plot, args.browser, level, args.path)
    else:
        for f in files:
            process_one(f, args.end_bin, args.plot, args.browser, args.level)

    return 0


if __name__ == '__main__':
    sys.exit(main())

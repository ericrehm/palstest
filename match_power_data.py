"""
Power Monitor Data Matching Script
Matches laser pulse timestamps to nearest power monitor measurements.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


# Some older shot files carry four min/max power columns instead of
# PowerASum/PowerBSum, and never have an associated PM file. The Min/Max
# values carry no usable information, so these files get PowerASum/PowerBSum
# seeded to 1.0 -- eqn 5's energy normalization (E_ref/E_L) becomes a no-op
# rather than dividing by a meaningless value, i.e. no pulse normalization.
OLD_POWER_COLUMNS = ('PowerAMin', 'PowerAMax', 'PowerBMin', 'PowerBMax')


def _is_old_power_format(header: list[str]) -> bool:
    return 'PowerASum' not in header and all(c in header for c in OLD_POWER_COLUMNS)


def load_laser_shots(filepath: str | Path) -> list[dict]:
    """
    Load laser shot data from CSV file.
    Returns list of records with Timestamp, PowerASum, PowerBSum, and other fields.
    """
    shots = []
    with open(filepath, 'r') as f:
        # Skip metadata lines starting with '#'
        header_line = None
        for line in f:
            if not line.startswith('#'):
                header_line = line.strip()
                break

        if not header_line:
            raise ValueError(f"No header found in {filepath}")

        # Parse header
        header = [col.strip() for col in header_line.split(',')]
        old_format = _is_old_power_format(header)

        # Read data
        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            if not row or not row.get('Timestamp'):
                continue

            try:
                timestamp = int(float(row.get('Timestamp', 0)))
                if old_format:
                    power_a_sum, power_b_sum = 1.0, 1.0
                else:
                    power_a_sum = float(row.get('PowerASum', 0))
                    power_b_sum = float(row.get('PowerBSum', 0))
                shots.append({
                    'timestamp': timestamp,
                    'identity': row.get('Identity', '').strip(),
                    'pitch': float(row.get('Pitch', 0)),
                    'power_a_sum_original': power_a_sum,
                    'power_b_sum_original': power_b_sum,
                    'laser_temp_c': float(row.get('LaserTempC', 0)),
                })
            except (ValueError, TypeError):
                continue

    return shots


def load_pm_data(filepath: str | Path, a_max_threshold: float = 100.0) -> list[dict]:
    """
    Load power monitor data, filter for A_Max > threshold, and group by timestamp.
    Returns list of unique timestamp records with max A_Max and B_Max at each timestamp.
    """
    # First pass: filter and group by timestamp
    pm_by_ts = {}
    
    with open(filepath, 'r') as f:
        # Skip metadata lines starting with '#'
        header_line = None
        for line in f:
            if not line.startswith('#'):
                header_line = line.strip()
                break
        
        if not header_line:
            raise ValueError(f"No header found in {filepath}")
        
        # Parse header
        header = [col.strip() for col in header_line.split(',')]
        
        # Read data and filter
        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            if not row or not row.get('Timestamp'):
                continue
            
            try:
                a_max = float(row.get('A_Max', 0))
                
                # Apply filter
                if a_max <= a_max_threshold:
                    continue
                
                timestamp = int(float(row.get('Timestamp', 0)))
                b_max = float(row.get('B_Max', 0))
                
                # Keep max values at each timestamp
                if timestamp not in pm_by_ts:
                    pm_by_ts[timestamp] = {
                        'timestamp': timestamp,
                        'a_max': a_max,
                        'b_max': b_max,
                    }
                else:
                    # Keep the maximum values at this timestamp
                    pm_by_ts[timestamp]['a_max'] = max(pm_by_ts[timestamp]['a_max'], a_max)
                    pm_by_ts[timestamp]['b_max'] = max(pm_by_ts[timestamp]['b_max'], b_max)
            except (ValueError, TypeError):
                continue
    
    # Return as sorted list by timestamp
    return sorted(pm_by_ts.values(), key=lambda x: x['timestamp'])


def find_nearest_pm_match(
    laser_timestamp: int,
    pm_records: list[dict],
    time_window_ms: float = 20.0,
    strict: bool = True
) -> Optional[dict]:
    """
    Find the nearest PM record (after filtering and aggregating by timestamp).
    
    Args:
        laser_timestamp: Timestamp of laser shot (ms)
        pm_records: Aggregated PM records (one per timestamp)
        time_window_ms: Maximum time difference allowed (default 20ms)
        strict: If True, enforce time_window_ms. If False, find absolute nearest regardless of window.
    
    Returns:
        Matched PM record, or None if no match found
    """
    closest = None
    closest_diff = float('inf')
    
    for record in pm_records:
        ts = record['timestamp']
        time_diff = abs(ts - laser_timestamp)
        
        # Check time constraint if strict mode
        if strict and time_diff > time_window_ms:
            continue
        
        # Keep the closest match
        if time_diff < closest_diff:
            closest = record
            closest_diff = time_diff
    
    return closest


def match_laser_to_pm(
    laser_shots: list[dict],
    pm_records: list[dict],
    time_window_ms: float = 20.0
) -> list[dict]:
    """
    Match each laser shot to nearest PM measurement.
    First tries strict matching (within time window), then fills unmatched with nearest neighbor.
    
    Returns list of matched records with both laser and PM data.
    """
    matched = []
    
    # First pass: strict matching with time window
    for shot in laser_shots:
        pm_match = find_nearest_pm_match(
            shot['timestamp'],
            pm_records,
            time_window_ms=time_window_ms,
            strict=True
        )
        
        result = {
            'timestamp': shot['timestamp'],
            'identity': shot['identity'],
            'pitch': shot['pitch'],
            'laser_temp_c': shot['laser_temp_c'],
            'power_a_sum_original': shot['power_a_sum_original'],
            'power_b_sum_original': shot['power_b_sum_original'],
            'pm_matched': pm_match is not None,
            'pm_matched_strict': pm_match is not None,  # Track strict matches
            'power_a_max': pm_match['a_max'] if pm_match else None,
            'power_b_max': pm_match['b_max'] if pm_match else None,
            'pm_timestamp': pm_match['timestamp'] if pm_match else None,
            'time_diff_ms': (pm_match['timestamp'] - shot['timestamp']) if pm_match else None,
        }
        matched.append(result)
    
    # Second pass: fill unmatched rows with nearest neighbor (relaxed matching)
    for result in matched:
        if not result['pm_matched']:
            # Use relaxed matching - find absolute nearest regardless of time window
            pm_match = find_nearest_pm_match(
                result['timestamp'],
                pm_records,
                time_window_ms=float('inf'),  # No time limit
                strict=False
            )
            
            if pm_match:
                result['pm_matched'] = True  # Mark as matched (but was fallback)
                result['power_a_max'] = pm_match['a_max']
                result['power_b_max'] = pm_match['b_max']
                result['pm_timestamp'] = pm_match['timestamp']
                result['time_diff_ms'] = pm_match['timestamp'] - result['timestamp']
    
    return matched


def plot_matched_data(matched_data: list[dict], output_file: str = 'power_matching_plot.html'):
    """
    Create interactive Plotly visualization of matched power data.
    Shows original vs matched values and time differences.
    """
    timestamps = [m['timestamp'] for m in matched_data]
    
    # Convert to relative time (seconds from first timestamp)
    if timestamps:
        first_ts = min(timestamps)
        rel_times = [(ts - first_ts) / 1000.0 for ts in timestamps]  # Convert to seconds
    else:
        rel_times = []
    
    original_a = [m['power_a_sum_original'] for m in matched_data]
    original_b = [m['power_b_sum_original'] for m in matched_data]
    matched_a = [m['power_a_max'] for m in matched_data]
    matched_b = [m['power_b_max'] for m in matched_data]
    time_diffs = [m['time_diff_ms'] if m['pm_matched'] else None for m in matched_data]
    matched_flags = [m['pm_matched'] for m in matched_data]
    
    # Create color array for time diff plot (only show matched points)
    time_diff_colors = []
    time_diff_x = []
    time_diff_y = []
    for i, (rel_t, m) in enumerate(zip(rel_times, matched_data)):
        if m['pm_matched']:
            time_diff_x.append(rel_t)
            time_diff_y.append(m['time_diff_ms'])
            time_diff_colors.append(abs(m['time_diff_ms']))
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Power A: Original vs Matched', 'Power B: Original vs Matched', 'PM Match Time Difference'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # Power A subplot
    fig.add_trace(
        go.Scatter(
            x=rel_times,
            y=original_a,
            mode='markers',
            name='PowerASum (Original)',
            marker=dict(size=5, color='blue'),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=rel_times,
            y=matched_a,
            mode='markers',
            name='A_Max (PM Matched)',
            marker=dict(size=5, color='cyan'),
        ),
        row=1, col=1
    )
    
    # Power B subplot
    fig.add_trace(
        go.Scatter(
            x=rel_times,
            y=original_b,
            mode='markers',
            name='PowerBSum (Original)',
            marker=dict(size=5, color='red'),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=rel_times,
            y=matched_b,
            mode='markers',
            name='B_Max (PM Matched)',
            marker=dict(size=5, color='orange'),
        ),
        row=2, col=1
    )
    
    # Time difference subplot - only show matched points
    fig.add_trace(
        go.Scatter(
            x=time_diff_x,
            y=time_diff_y,
            mode='markers',
            name='Time Diff (ms)',
            marker=dict(
                size=8,
                color=time_diff_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Diff (ms)", len=0.3, y=0.16),
            ),
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (seconds from start)", row=3, col=1)
    fig.update_yaxes(title_text="Power A", row=1, col=1)
    fig.update_yaxes(title_text="Power B", row=2, col=1)
    fig.update_yaxes(title_text="Time Diff (ms)", row=3, col=1)
    
    fig.update_layout(
        height=1000,
        title_text="Laser Pulse Power Matching: Original vs PM Data",
        hovermode='x unified',
        showlegend=True,
    )
    
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")
    
    return fig


def print_match_summary(matched_data: list[dict]):
    """Print summary statistics of the matching results."""
    total = len(matched_data)
    strict_matched = sum(1 for m in matched_data if m.get('pm_matched_strict', False))
    fallback_matched = sum(1 for m in matched_data if m['pm_matched'] and not m.get('pm_matched_strict', False))
    total_matched = strict_matched + fallback_matched
    
    print(f"\n{'='*60}")
    print(f"POWER MATCHING SUMMARY")
    print(f"{'='*60}")
    print(f"Total laser shots: {total}")
    print(f"Strict matches (within ±20ms): {strict_matched} ({100*strict_matched/total:.1f}%)")
    print(f"Fallback matches (nearest neighbor): {fallback_matched} ({100*fallback_matched/total:.1f}%)")
    print(f"Total matched: {total_matched} ({100*total_matched/total:.1f}%)")
    print(f"Unmatched: {total - total_matched}")
    
    if strict_matched > 0:
        strict_diffs = [abs(m['time_diff_ms']) for m in matched_data if m.get('pm_matched_strict', False)]
        print(f"\nStrict match time difference statistics:")
        print(f"  Min: {min(strict_diffs):.2f} ms")
        print(f"  Max: {max(strict_diffs):.2f} ms")
        print(f"  Mean: {sum(strict_diffs)/len(strict_diffs):.2f} ms")
    
    if total_matched > 0:
        # Sample matches
        print(f"\nFirst 5 matches:")
        count = 0
        for i, m in enumerate(matched_data):
            if m['pm_matched']:
                match_type = "strict" if m.get('pm_matched_strict', False) else "fallback"
                print(f"  Shot {i} ({match_type}): A_Max={m['power_a_max']:.2f}, B_Max={m['power_b_max']:.2f}, "
                      f"Time diff={m['time_diff_ms']:.2f}ms")
                count += 1
                if count >= 5:
                    break
    
    print(f"{'='*60}\n")


def export_matched_csv(
    laser_file: str | Path,
    matched_data: list[dict],
    output_file: str = 'laser_with_pm_power.csv'
) -> None:
    """
    Export matched data to CSV file matching original laser file structure.
    Replaces PowerASum and PowerBSum with matched A_Max and B_Max values.
    
    Args:
        laser_file: Path to original laser CSV file
        matched_data: List of matched records from match_laser_to_pm()
        output_file: Path to write output CSV
    """
    # Create lookup dict for quick access
    matched_by_ts = {m['timestamp']: m for m in matched_data}

    with open(laser_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Preserve original metadata lines, then read header.
        # L0 and L0b share the same file format; only PowerASum/PowerBSum values differ.
        metadata_lines = []
        header_line = None
        for line in infile:
            if line.startswith('#'):
                metadata_lines.append(line.rstrip('\n'))
            else:
                header_line = line.strip()
                break

        if not header_line:
            raise ValueError(f"No header found in {laser_file}")

        # Parse header to find column indices
        header = [col.strip() for col in header_line.split(',')]
        old_format = _is_old_power_format(header)

        if old_format:
            # Drop the four useless Min/Max columns and write the standard
            # PowerASum/PowerBSum pair instead, in their place, so every L0b
            # file downstream has the same schema regardless of source format.
            insert_at = header.index('PowerAMin')
            out_header = [c for c in header if c not in OLD_POWER_COLUMNS]
            out_header[insert_at:insert_at] = ['PowerASum', 'PowerBSum']
        else:
            out_header = header

        # Write original metadata forward, then an L0b provenance line, then header
        for line in metadata_lines:
            outfile.write(line + '\n')
        outfile.write(f"# L0b processed: {datetime.now(timezone.utc).isoformat()}\n")
        outfile.write((header_line if not old_format else ','.join(out_header)) + '\n')

        # Process data rows
        reader = csv.DictReader(infile, fieldnames=header)
        writer = csv.DictWriter(outfile, fieldnames=out_header)

        rows_written = 0
        rows_with_pm = 0

        for row in reader:
            if not row or not row.get('Timestamp'):
                continue

            try:
                timestamp = int(float(row.get('Timestamp', 0)))

                if old_format:
                    row = {k: v for k, v in row.items() if k not in OLD_POWER_COLUMNS}
                    row['PowerASum'] = '1.0'
                    row['PowerBSum'] = '1.0'

                # Check if we have a match for this timestamp
                if timestamp in matched_by_ts:
                    match = matched_by_ts[timestamp]
                    if match['pm_matched']:
                        # Replace with PM values
                        row['PowerASum'] = str(match['power_a_max'])
                        row['PowerBSum'] = str(match['power_b_max'])
                        rows_with_pm += 1
                    # If not matched, keep original (or seeded 1.0) values

                writer.writerow(row)
                rows_written += 1
            except (ValueError, TypeError):
                continue
    
    print(f"\nExported matched data to {output_file}")
    print(f"  Total rows written: {rows_written}")
    print(f"  Rows with PM power values: {rows_with_pm}")


def main():
    """Main execution - load data, match, and visualize."""
    # File paths
    laser_file = 'cruisedata/D3_1/D3_1(D) transit_3.csv'
    pm_file = 'cruisedata/D3_1/D3_1(D) transit_PM_3.csv'
    output_csv = 'cruisedata/D3_1/D3_1(D) transit_3_with_PM_power.csv'
    
    print("Loading laser shot data...")
    laser_shots = load_laser_shots(laser_file)
    print(f"  Loaded {len(laser_shots)} laser shots")
    
    print("Loading and filtering power monitor data (A_Max > 100)...")
    pm_records = load_pm_data(pm_file, a_max_threshold=100.0)
    print(f"  Loaded and aggregated to {len(pm_records)} unique timestamps")
    
    print("Matching laser shots to PM data (±20ms window)...")
    matched_data = match_laser_to_pm(laser_shots, pm_records, time_window_ms=20.0)
    
    print_match_summary(matched_data)
    
    print("Creating visualization...")
    plot_matched_data(matched_data, output_file='power_matching_plot.html')
    
    print("Exporting matched data to CSV...")
    export_matched_csv(laser_file, matched_data, output_file=output_csv)
    
    return matched_data


if __name__ == '__main__':
    matched_results = main()

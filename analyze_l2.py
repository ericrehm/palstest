"""
Per-channel summary of one or more L2 files: mean and standard deviation of
KLidar/CT532Near/CT532Far/CT650Near/CT650Far across all shots, one row per
natural identity (co_near/cross_near/raman_near/co_far/cross_far/raman_far
-- the six real detector channels, not L1CrossProcessor's derived
qer_/depol_ rows or the ensemble-SNR synthetic rows). Given more than one
file, every file's rows are concatenated first -- the mean/stdev are taken
across the combined set, not averaged per-file and then re-averaged.

Usage:
    python3 analyze_l2.py path/to/some_L2_0.csv [-o output.csv]
    python3 analyze_l2.py path/to/*_L2_*.csv [-o output.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NATURAL_IDENTITIES = ['co_near', 'cross_near', 'raman_near', 'co_far', 'cross_far', 'raman_far']
METRICS = ['KLidar', 'CT532Near', 'CT532Far', 'CT650Near', 'CT650Far']


def _header_row(path: Path) -> int:
    """0-indexed line number of the CSV header row -- an L2 file (see
    v2.io.pals_io.write_l1_shots) starts with '#' metadata comment lines,
    then exactly one header row, before the data."""
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.startswith('#'):
                return i
    raise ValueError(f"No header found in {path}")


def load_l2_summary_columns(path: Path) -> pd.DataFrame:
    """Timestamp/Identity plus the METRICS columns only -- an L2 file's
    Bin_/Range_/STILDE_/QA*_ columns can run into hundreds of MB for a real
    cruise file, and none of them are needed here, so usecols keeps pandas
    from ever reading them off disk in the first place.
    """
    header_row = _header_row(path)
    wanted = ['Timestamp', 'Identity'] + METRICS

    # Peeked at separately (nrows=0, just the header) rather than letting
    # `usecols` do the validation: pandas raises its own much less
    # readable ValueError straight out of read_csv when a requested column
    # isn't in the file at all (e.g. an older L1-only file with no
    # CT532Near/etc, or KLidar itself missing on a raw/L0b file), which is
    # exactly the common misuse this should explain clearly instead.
    available = pd.read_csv(path, skiprows=header_row, nrows=0).columns
    missing = [c for c in wanted if c not in available]
    if missing:
        raise ValueError(
            f"{path.name} is missing expected column(s) {missing} -- "
            f"is this an L2 file (not raw/L0b/L1)?"
        )

    df = pd.read_csv(path, skiprows=header_row, usecols=wanted)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean_table, stdev_table), each indexed by NATURAL_IDENTITIES (in
    that fixed order, even for a channel with zero matching rows -- filled
    with NaN rather than silently missing a row) with METRICS as columns.
    mean/std both skip NaN by default (blank KLidar on a QAK=MISSING shot,
    e.g.), same "exclude, don't zero-fill" convention used everywhere else
    in this pipeline's averaging.
    """
    natural = df[df['Identity'].isin(NATURAL_IDENTITIES)]
    grouped = natural.groupby('Identity')[METRICS]
    means = grouped.mean().reindex(NATURAL_IDENTITIES)
    means.index.name = 'Identity'
    stds = grouped.std().reindex(NATURAL_IDENTITIES)
    stds.index.name = 'Identity'
    return means, stds


def write_summary_csv(means: pd.DataFrame, stds: pd.DataFrame, out_path: Path) -> None:
    """Both tables in one CSV -- a labeled block per table, one blank line
    between them. Still just one file, so it's the natural single artifact
    to hand someone, but each block reads on its own as an ordinary
    Identity-by-METRICS table if pulled into a spreadsheet or re-parsed."""
    with open(out_path, 'w', newline='') as f:
        f.write('mean\n')
        means.to_csv(f)
        f.write('\nstdev\n')
        stds.to_csv(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('l2_files', type=Path, nargs='+', help='Path to one or more L2 CSV files')
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help='Output CSV path (default: <first file\'s stem>_summary.csv next to it)',
    )
    args = parser.parse_args()

    for path in args.l2_files:
        if not path.is_file():
            parser.error(f'No such file: {path}')

    try:
        parts = [load_l2_summary_columns(path) for path in args.l2_files]
    except ValueError as e:
        parser.error(str(e))
    # Concatenated before summarizing, not summarized per-file and
    # re-averaged -- mean/stdev of the combined row set, per the ask.
    df = pd.concat(parts, ignore_index=True)
    means, stds = summarize(df)

    out_path = args.output or args.l2_files[0].with_name(f'{args.l2_files[0].stem}_summary.csv')
    write_summary_csv(means, stds, out_path)

    if len(args.l2_files) == 1:
        print(f'{len(df)} rows loaded from {args.l2_files[0].name}')
    else:
        print(f'{len(df)} rows loaded from {len(args.l2_files)} files:')
        for path, part in zip(args.l2_files, parts):
            print(f'  {len(part)} rows from {path.name}')
    print()
    print('mean:')
    print(means.to_string())
    print()
    print('stdev:')
    print(stds.to_string())
    print()
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()

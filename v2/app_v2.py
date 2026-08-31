"""
Flask application wrapper for PALS V2 processing pipeline.
Thin client over Python processing layer.
"""

from flask import Flask, jsonify, request, render_template, Response
from pathlib import Path
from typing import Dict, Optional
import dataclasses
import io
import json
import re
import sys
import tempfile

# Add parent directory to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from v2.processing import L0bMatcher, L1Processor, L1CrossProcessor, L2Processor, L3Processor
from v2.io import (
    read_l0_from_csv,
    parse_l0b_shots,
    parse_l1_shots,
    write_l0b_to_csv,
    write_l1_shots,
    write_l1_to_csv,
    write_l2_to_csv,
    write_l3_to_csv,
)
from v2.config import load_config
from v2.viewer_data import ViewerFile, load_raw_or_l0b, load_l1_or_l2

# The instrument's shot-file naming convention: subsequent files in a
# sequence get "_N" appended, but the FIRST file is written with no numeric
# suffix at all (e.g. "foo.csv" then "foo_1.csv", "foo_2.csv", ...). PM
# sibling files follow the same rule, with "_PM" inserted right before the
# sequence number (or appended at the end when there is none).
_SEQUENCE_SUFFIX_RE = re.compile(r'_(\d+)$')


app = Flask(__name__, template_folder="templates", static_folder="static")

# Configuration
CONFIG_FILE = Path(__file__).parent / "config" / "PALS_SBS312.json"
if CONFIG_FILE.exists():
    INSTRUMENT_CONFIG = load_config(CONFIG_FILE)
else:
    INSTRUMENT_CONFIG = {"instrument_id": "PALS_UNKNOWN", "channels": {}}

# Initialize processors
L0B_MATCHER = L0bMatcher(fallback_energy=1.0)
L1_PROC = L1Processor(INSTRUMENT_CONFIG)
L1_CROSS_PROC = L1CrossProcessor()  # stateless: no config needed
L2_PROC = L2Processor(INSTRUMENT_CONFIG)
L3_PROC = L3Processor(INSTRUMENT_CONFIG)

# Single "currently loaded file" slot for the ad-hoc viewer (see
# viewer_data.py) -- a local inspection tool, not multi-user state.
VIEWER_FILE: Optional[ViewerFile] = None

# Two independent slots for the file-vs-file compare page -- separate from
# VIEWER_FILE above so the (already fiddly to get right) single-file viewer
# is untouched by this.
COMPARE_FILES: Dict[str, Optional[ViewerFile]] = {"a": None, "b": None}


@app.route("/")
def index():
    """Main page"""
    return render_template(
        "index_v2.html", instrument=INSTRUMENT_CONFIG.get("instrument_id", "?"), active_tab="processing"
    )


@app.route("/viewer")
def viewer_page():
    """Ad-hoc raw/L0b/L1/L2 heatmap+waveform viewer"""
    return render_template(
        "viewer.html", instrument=INSTRUMENT_CONFIG.get("instrument_id", "?"), active_tab="viewer1"
    )


@app.route("/compare")
def compare_page():
    """File-vs-file waveform comparator ('visual diff')"""
    return render_template(
        "compare.html", instrument=INSTRUMENT_CONFIG.get("instrument_id", "?"), active_tab="viewer2"
    )


@app.route("/api/viewer/load", methods=["POST"])
def viewer_load():
    """
    Parse an uploaded file once and cache it server-side as VIEWER_FILE.
    Returns metadata only (identity list, time range, per-identity bin/range
    axis) -- the actual signal matrices are fetched a 20s window at a time via
    /api/viewer/window, never all at once (see viewer_data.py's docstring for
    why: a whole L1/L2 cruise file's matrix is hundreds of MB as JSON).
    """
    global VIEWER_FILE

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    filename = file.filename or "uploaded.csv"
    file_type = request.form.get('fileType', 'raw')  # 'raw' | 'l0b' | 'l1' | 'l2'

    try:
        content = file.read().decode('utf-8-sig', errors='replace')
        lines = content.splitlines(keepends=True)
        if file_type in ('raw', 'l0b'):
            viewer_file = load_raw_or_l0b(lines, filename)
        else:
            viewer_file = load_l1_or_l2(lines, filename)
        viewer_file.file_type = file_type
        VIEWER_FILE = viewer_file
        return jsonify(viewer_file.meta()), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load file for viewing: {e}"}), 500


@app.route("/api/viewer/window", methods=["GET"])
def viewer_window():
    """Return one identity's (timestamps, z-matrix) for a single time slice."""
    if VIEWER_FILE is None:
        return jsonify({"error": "No file loaded"}), 400
    identity = request.args.get('identity', '')
    try:
        start_ms = int(request.args.get('start_ms', '0'))
        end_ms = int(request.args.get('end_ms', '0'))
    except ValueError:
        return jsonify({"error": "start_ms/end_ms must be integers"}), 400
    return jsonify(VIEWER_FILE.window(identity, start_ms, end_ms)), 200


@app.route("/api/compare/load", methods=["POST"])
def compare_load():
    """Same parsing as /api/viewer/load, but into one of two independent
    slots ('a' or 'b') so two files can be loaded and inspected at once."""
    slot = request.form.get('slot', 'a')
    if slot not in COMPARE_FILES:
        return jsonify({"error": f"Invalid slot: {slot!r}"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    filename = file.filename or "uploaded.csv"
    file_type = request.form.get('fileType', 'raw')

    try:
        content = file.read().decode('utf-8-sig', errors='replace')
        lines = content.splitlines(keepends=True)
        if file_type in ('raw', 'l0b'):
            viewer_file = load_raw_or_l0b(lines, filename)
        else:
            viewer_file = load_l1_or_l2(lines, filename)
        viewer_file.file_type = file_type
        COMPARE_FILES[slot] = viewer_file
        return jsonify(viewer_file.meta()), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load file for viewing: {e}"}), 500


@app.route("/api/compare/window", methods=["GET"])
def compare_window():
    """Return one slot's (timestamps, z-matrix) for a single time slice --
    same shape as /api/viewer/window, just scoped to a slot."""
    slot = request.args.get('slot', 'a')
    viewer_file = COMPARE_FILES.get(slot)
    if viewer_file is None:
        return jsonify({"error": f"No file loaded in slot {slot!r}"}), 400
    identity = request.args.get('identity', '')
    try:
        start_ms = int(request.args.get('start_ms', '0'))
        end_ms = int(request.args.get('end_ms', '0'))
    except ValueError:
        return jsonify({"error": "start_ms/end_ms must be integers"}), 400
    return jsonify(viewer_file.window(identity, start_ms, end_ms)), 200


@app.route("/api/compare/shot", methods=["GET"])
def compare_shot():
    """Single shot's waveform nearest to target_ms, for one slot/identity."""
    slot = request.args.get('slot', 'a')
    viewer_file = COMPARE_FILES.get(slot)
    if viewer_file is None:
        return jsonify({"error": f"No file loaded in slot {slot!r}"}), 400
    identity = request.args.get('identity', '')
    try:
        target_ms = int(request.args.get('target_ms', '0'))
    except ValueError:
        return jsonify({"error": "target_ms must be an integer"}), 400
    return jsonify(viewer_file.nearest(identity, target_ms)), 200


@app.route("/api/compare/average", methods=["GET"])
def compare_average():
    """Ensemble-average waveform for one identity in one slot -- a single
    waveform (n_bins floats), computed server-side from the already-resident
    z-matrix rather than shipping every shot to the browser to average."""
    slot = request.args.get('slot', 'a')
    viewer_file = COMPARE_FILES.get(slot)
    if viewer_file is None:
        return jsonify({"error": f"No file loaded in slot {slot!r}"}), 400
    identity = request.args.get('identity', '')
    return jsonify(viewer_file.average(identity)), 200


# Disabled: not called by the current UI (index_v2.html only uses
# /api/process-file). Relies on read_l0_from_csv() (only reads row 0 of a
# shot file, doesn't understand the real Identity-per-row format) and
# L0B_MATCHER.process() (a stub that always returns hardcoded 42.5/3.2
# regardless of input). Also now out of date: L1Processor.process() requires
# an e_ref argument (mean PowerASum across the file) that this route never
# computes. Commented out rather than fixed, since fixing it means first
# fixing read_l0_from_csv/L0bMatcher for the real file format -- a separate
# task from L1's eqn 1/4/5 work.
#
# @app.route("/api/process", methods=["POST"])
# def process_l0():
#     """
#     Accept L0 file(s) and return processing products.
#
#     Expected JSON:
#     {
#         "l0_file": "path/to/shot.csv" or "filename.csv" (searches testdata/ and cruisedata/),
#         "pm_dir": "path/to/pm/data",  # Optional
#         "levels": ["l1", "l2", "l3"]  # Which levels to compute
#     }
#     """
#     payload = request.get_json(silent=True) or {}
#     l0_file_input = payload.get("l0_file")
#     pm_dir = payload.get("pm_dir")
#     levels = payload.get("levels", ["l1"])
#
#     if not l0_file_input:
#         return jsonify({"error": "Missing l0_file"}), 400
#
#     # Resolve file path: try direct path, then relative to project root, then search common directories
#     l0_file = Path(l0_file_input)
#
#     # Try as absolute/direct path first
#     if not l0_file.exists():
#         # Try relative to project root
#         project_root = Path(__file__).parent.parent
#         l0_file_from_root = project_root / l0_file_input
#         if l0_file_from_root.exists():
#             l0_file = l0_file_from_root
#         else:
#             # Search in common directories (for bare filenames)
#             search_dirs = [
#                 Path.cwd() / "testdata",
#                 Path.cwd() / "cruisedata",
#                 project_root / "testdata",
#                 project_root / "cruisedata",
#             ]
#             found = False
#             for search_dir in search_dirs:
#                 candidate = search_dir / l0_file_input
#                 if candidate.exists():
#                     l0_file = candidate
#                     found = True
#                     break
#
#             if not found:
#                 return jsonify({"error": f"File not found: {l0_file_input}. Searched in: {[str(d) for d in search_dirs]}"}), 404
#
#     try:
#         # Read L0
#         l0 = read_l0_from_csv(l0_file)
#
#         # Produce L0b
#         l0b = L0B_MATCHER.process(l0)
#
#         # Write L0b output file to Processed subdirectory in the input file's directory
#         output_dir = l0_file.parent / "Processed"
#         output_dir.mkdir(exist_ok=True)
#         l0b_output_path = output_dir / f"{l0_file.stem}_L0b_{l0b.l0.sequence_id}.csv"
#         write_l0b_to_csv(l0b, l0b_output_path)
#
#         results = {
#             "instrument_id": INSTRUMENT_CONFIG.get("instrument_id"),
#             "l0_file": str(l0_file),
#             "l0b_output_file": str(l0b_output_path),
#             "l0": l0.to_dict(),
#             "l0b": l0b.to_dict(),
#         }
#
#         # L1 processing
#         if "l1" in levels:
#             l1 = L1_PROC.process(l0b)
#             results["l1_available"] = True
#             results["l1_channels"] = list(l1.signal.keys())
#
#         # L2 processing (requires ensemble)
#         if "l2" in levels:
#             # Placeholder: use single L1 as ensemble
#             l2 = L2_PROC.process([l1])
#             results["l2_available"] = True
#             results["l2_range_m"] = l2.range_m.tolist()
#
#         # L3 processing
#         if "l3" in levels:
#             l3 = L3_PROC.process(l2)
#             results["l3_available"] = True
#
#         return jsonify(results), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def get_config():
    """Return current instrument configuration"""
    return jsonify(INSTRUMENT_CONFIG), 200


@app.route("/api/export/<level>/<channel>", methods=["GET"])
def export_product(level: str, channel: str):
    """
    Export a product level to CSV.
    (Implementation depends on persisting products; placeholder here)
    """
    return jsonify({"error": "Not yet implemented"}), 501


@app.route("/api/process-file", methods=["POST"])
def process_file():
    """
    Accept an uploaded file and run exactly ONE pipeline stage on it, returning
    the result as a raw text/csv response body (metadata goes in headers, not
    embedded in JSON). There is no separate Processed/ subfolder -- pipeline
    outputs are written by the browser directly alongside the raw file, in
    whatever directory the user picked.

    One stage per call (not "levels" -- rewritten from the earlier design)
    because bundling L0b+L1+L2 into a single JSON response doesn't scale: a
    real cruise file's combined L1+L2 CSV content, embedded as JSON string
    fields, was projected at >1GB for a single input file. The browser now
    chains requests itself (see index_v2.html), writing/streaming each
    stage's raw CSV response to disk via response.blob() before uploading it
    as the next stage's input.

    Form fields:
      - 'file': the upload. Its *meaning* is given by 'inputStage', not by
        which pipeline stage is being requested.
      - 'stage': which single stage to produce -- 'l0b', 'l1', or 'l2'.
        ('l3' is accepted but not implemented; returns a warning, no output.)
      - 'inputStage': what the uploaded bytes already are -- 'raw' (a raw L0
        shot file; required for stage='l0b'), 'l0b' (required for
        stage='l1'), or 'l1' (required for stage='l2').
      - 'originalFilename': the RAW file's name, always -- used for output
        naming and to locate the raw file (and its PM sibling) on disk when
        deriving L0b, regardless of which stage is actually being requested.
      - 'sourceDir': directory name the file was picked from, so the server
        can find it (and, for L0b, its PM sibling) on its own filesystem.
      - 'pmtModelSource', 'preTriggerBins': L1-only UI options.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    original_filename = request.form.get('originalFilename') or file.filename or ""

    if not file or not file.filename or not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400

    stage = request.form.get('stage')
    input_stage = request.form.get('inputStage', 'raw')
    source_dir_name = request.form.get('sourceDir')  # Directory name where file was selected from
    pmt_model_source = request.form.get('pmtModelSource', 'datasheet')  # L1 UI option, not a config setting
    pre_trigger_bins_raw = request.form.get('preTriggerBins')  # L1 UI option; None -> config default
    pre_trigger_bins = int(pre_trigger_bins_raw) if pre_trigger_bins_raw not in (None, '') else None

    # The instrument's naming convention (see _SEQUENCE_SUFFIX_RE) makes the
    # trailing "_N" *be* the sequence id, with the first file in a sequence
    # implicitly "0" (no suffix at all). Deriving it from the filename rather
    # than hardcoding 0 matters once a directory holds more than one shot
    # file: hardcoding 0 would make every file's outputs collide on the same
    # "..._L0b_0.csv" name.
    seq_match = _SEQUENCE_SUFFIX_RE.search(Path(original_filename).stem)
    sequence_id = int(seq_match.group(1)) if seq_match else 0

    if stage == "l3":
        return jsonify({"warnings": ["L3 processing not yet implemented in this build"]}), 200

    required_input_stage = {"l0b": "raw", "l1": "l0b", "l2": "l1"}.get(stage or "")
    if required_input_stage is None:
        return jsonify({"error": f"Unsupported or missing stage: {stage!r}"}), 400
    if input_stage != required_input_stage:
        return jsonify({
            "error": (
                f"Stage '{stage}' requires inputStage='{required_input_stage}', "
                f"got '{input_stage}'"
            )
        }), 400

    try:
        if stage == "l0b":
            project_root = Path(__file__).parent.parent

            # Find the original shot file on the server's filesystem -- needed
            # to locate its sibling PM file (match_power_data reads both from
            # disk).
            original_file = None
            if source_dir_name:
                search_paths = [
                    project_root / "cruisedata" / source_dir_name / original_filename,
                    project_root / "testdata" / source_dir_name / original_filename,
                    project_root / source_dir_name / original_filename,
                ]
                for candidate_path in search_paths:
                    if candidate_path.exists():
                        original_file = candidate_path
                        break

            if not original_file:
                return jsonify({"error": f"Could not find original file: {original_filename}"}), 400

            sys.path.insert(0, str(project_root))
            from match_power_data import load_laser_shots, load_pm_data, match_laser_to_pm, export_matched_csv

            laser_shots = load_laser_shots(original_file)

            # Exact-match PM sibling lookup, not a glob search. A glob (e.g.
            # "*PM*.csv") is ambiguous the moment a directory holds more than
            # one shot+PM pair -- exactly the case that broke the no-sequence-
            # number first file, since its PM sibling ("..._PM.csv") and a
            # later file's PM sibling ("..._PM_1.csv") would both match the
            # same glob. The instrument's convention inserts "_PM" right
            # before the sequence number (or appends it, when there is none).
            m = _SEQUENCE_SUFFIX_RE.search(original_file.stem)
            if m:
                pm_stem = original_file.stem[:m.start()] + '_PM_' + m.group(1)
            else:
                pm_stem = original_file.stem + '_PM'
            pm_file = original_file.with_name(pm_stem + original_file.suffix)

            pm_records = []
            if pm_file.exists():
                try:
                    pm_records = load_pm_data(pm_file)
                except Exception:
                    pm_records = []

            # match_laser_to_pm() handles an empty pm_records list correctly
            # on its own -- every shot comes back properly shaped with
            # pm_matched=False and its original PowerASum/PowerBSum preserved
            # -- so always route through it rather than short-circuiting to
            # the raw (differently-shaped) laser_shots.
            matched_data = match_laser_to_pm(laser_shots, pm_records)
            pm_matched = bool(pm_records)

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
                tmp_l0b_path = tmp.name
            export_matched_csv(original_file, matched_data, tmp_l0b_path)
            with open(tmp_l0b_path, 'r') as f:
                l0b_content = f.read()
            Path(tmp_l0b_path).unlink(missing_ok=True)

            return Response(l0b_content, mimetype='text/csv', headers={
                'X-Sequence-Id': str(sequence_id),
                'X-PM-Matched': 'true' if pm_matched else 'false',
            })

        elif stage == "l1":
            content = file.read().decode('utf-8')
            l0b_shots = parse_l0b_shots(content.splitlines(keepends=True), source_file=original_filename or "")
            if not l0b_shots:
                return jsonify({"warnings": ["L1 skipped: no recognized shots found in L0b input"]}), 200

            # Eqn 5's E_ref: mean PowerASum across all shots in this file
            e_ref = sum(s.power_a_sum for s in l0b_shots) / len(l0b_shots)

            # pmt_model_source/pre_trigger_bins are per-request UI choices,
            # not config settings -- reuse the shared default processor when
            # both match, otherwise build one for this request's choices.
            if pmt_model_source == L1_PROC.pmt_model_source and (
                pre_trigger_bins is None or pre_trigger_bins == L1_PROC.pre_trigger_bins
            ):
                l1_proc = L1_PROC
            else:
                l1_proc = L1Processor(
                    INSTRUMENT_CONFIG, pmt_model_source=pmt_model_source, pre_trigger_bins=pre_trigger_bins
                )
            l1_shots = [l1_proc.process(shot, e_ref) for shot in l0b_shots]

            # Cross-channel diagnostics (section 6.3: Q_ER, depolarization)
            # need two different channels' shots at once, which L1Processor
            # can't see one-shot-at-a-time -- L1CrossProcessor runs as a
            # second pass over the whole file's L1 shots and emits new
            # synthetic-Identity rows (qer_near/qer_far/depol_near/depol_far)
            # written into the same file.
            cross_shots = L1_CROSS_PROC.process(l1_shots)

            l1_buf = io.StringIO()
            write_l1_shots(l1_shots + cross_shots, l1_buf)

            return Response(l1_buf.getvalue(), mimetype='text/csv', headers={
                'X-Sequence-Id': str(sequence_id),
                'X-PMT-Model-Source': pmt_model_source,
                'X-Pre-Trigger-Bins': str(l1_proc.pre_trigger_bins),
            })

        else:  # stage == "l2"
            content = file.read().decode('utf-8')
            l1_shots_all = parse_l1_shots(content.splitlines(keepends=True), source_file=original_filename or "")
            if not l1_shots_all:
                return jsonify({"warnings": ["L2 skipped: no recognized shots found in L1 input"]}), 200

            by_channel: Dict[str, list] = {}
            for shot in l1_shots_all:
                channel_id = next(iter(shot.signal))
                by_channel.setdefault(channel_id, []).append(shot)

            # Per section 5.5: range correction (eqn 7) is a per-shot
            # operation -- L2 shots are just L1 shots with X_j(r) = r^2 *
            # S_hat_j applied (see L2Processor.process) -- so the main L2
            # file has the same row-per-shot shape as L1's. Ensemble SNR
            # (mean(S-tilde)/std(S-tilde) across all same-identity shots) is
            # the one genuinely ensemble-dependent product; it's written as
            # one extra synthetic-Identity row per channel, not per-shot,
            # alongside the per-shot rows in the same file.
            l2_shots = []
            ensemble_snr_shots = []
            warnings = []
            for channel_id, ensemble in by_channel.items():
                try:
                    l2 = L2_PROC.process(ensemble)
                except Exception as e:
                    warnings.append(f"L2 skipped for {channel_id}: {e}")
                    continue
                for l1_shot, rc in zip(l2.l1_ensemble, l2.range_corrected):
                    l2_shots.append(dataclasses.replace(l1_shot, signal=rc))
                ensemble_snr_shots.append(L2_PROC.build_ensemble_snr(ensemble, channel_id))

            if not l2_shots:
                warnings = warnings or ["L2 skipped: no channel ensembles available"]
                return jsonify({"warnings": warnings}), 200

            l2_buf = io.StringIO()
            write_l1_shots(l2_shots + ensemble_snr_shots, l2_buf, stage_label="L2")

            headers = {'X-Sequence-Id': str(sequence_id)}
            if warnings:
                headers['X-Warnings'] = json.dumps(warnings)
            return Response(l2_buf.getvalue(), mimetype='text/csv', headers=headers)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Port 5000 is claimed by macOS's AirPlay Receiver (ControlCenter) on most Macs,
    # which causes requests to be flakily routed between it and this server. 5025
    # avoids that collision.
    app.run(debug=True, port=5025)

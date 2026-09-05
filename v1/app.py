from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from pypals import PalsData, default_data_path


app = Flask(__name__)
data_store = PalsData(default_data_path())


def _normalize_user_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _discover_csv_files(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() != ".csv":
            raise ValueError("Selected file is not a .csv")
        return [target]

    if target.is_dir():
        files = sorted(p for p in target.glob("*.csv") if p.is_file())
        if not files:
            raise ValueError("Directory has no .csv files")
        return files

    raise ValueError("Path does not exist")


@app.route("/")
def index() -> str:
    return render_template("index.html", app_root=str(Path.cwd()))


@app.route("/api/metadata")
def metadata() -> tuple:
    return jsonify(data_store.get_metadata_payload())


@app.route("/api/heatmaps")
def heatmaps() -> tuple:
    sigma_v = request.args.get("sigma_v", type=float, default=1.5)
    sigma_h = request.args.get("sigma_h", type=float, default=4.0)
    return jsonify(data_store.get_heatmaps_payload(sigma_v=sigma_v, sigma_h=sigma_h))


@app.route("/api/waveforms")
def waveforms() -> tuple:
    timestamp = request.args.get("timestamp", type=int)
    if timestamp is None:
        return jsonify({"error": "Query parameter 'timestamp' is required"}), 400
    
    sigma_v = request.args.get("sigma_v", type=float, default=0)
    sigma_h = request.args.get("sigma_h", type=float, default=0)

    return jsonify(data_store.get_waveforms_payload(timestamp, sigma_v=sigma_v, sigma_h=sigma_h))


@app.route("/api/power")
def power() -> tuple:
    identity = request.args.get("identity", type=str)
    return jsonify(data_store.get_power_payload(identity=identity))


@app.route("/api/laser-temp")
def laser_temp() -> tuple:
    identity = request.args.get("identity", type=str)
    return jsonify(data_store.get_laser_temp_payload(identity=identity))


@app.route("/api/attitude")
def attitude() -> tuple:
    identity = request.args.get("identity", type=str)
    return jsonify(data_store.get_attitude_payload(identity=identity))


@app.route("/api/load-path", methods=["POST"])
def load_path() -> tuple:
    global data_store

    payload = request.get_json(silent=True) or {}
    raw_path = str(payload.get("path", "")).strip()
    if not raw_path:
        return jsonify({"error": "JSON body must include 'path'"}), 400

    try:
        normalized = _normalize_user_path(raw_path)
        csv_files = _discover_csv_files(normalized)
        data_store = PalsData(csv_files)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            # data_store.loaded_files, not [str(path) for path in csv_files]
            # -- a file with no header/data is skipped rather than failing
            # the whole batch (see PalsData._load), so this must reflect
            # what actually loaded, not everything discovered on disk.
            "loaded_files": data_store.loaded_files,
            "metadata": data_store.get_metadata_payload(),
        }
    )


@app.route("/api/load-files", methods=["POST"])
def load_files() -> tuple:
    global data_store

    uploaded_files = request.files.getlist("files")
    valid_files = [f for f in uploaded_files if f and (f.filename or "").strip()]
    if not valid_files:
        return jsonify({"error": "No files were uploaded"}), 400

    try:
        data_store = PalsData.from_uploaded_files(valid_files)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            # data_store.loaded_files, not [f.filename for f in valid_files]
            # -- a file with no header/data is skipped rather than failing
            # the whole batch (see PalsData.from_uploaded_files), so this
            # must reflect what actually loaded, not everything uploaded.
            "loaded_files": data_store.loaded_files,
            "metadata": data_store.get_metadata_payload(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)

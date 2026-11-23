"""Historical AIS receiver simulation and short audio preview API."""

from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path

import soundfile as sf
from flask import Blueprint, current_app, jsonify, request, send_file

from data_io.ais import load_ais, parse_utc
from logging_utils.simulation_log import save_preview_run
from sim.audio_mix import mix_preview
from sim.propagation import level_series, load_settings, scene_at, thorp_db_per_km


simulator_bp = Blueprint("simulator_data", __name__, url_prefix="/api/simulator")


@simulator_bp.errorhandler(ValueError)
def bad_request(error):
    return jsonify(error=str(error)), 400


@simulator_bp.errorhandler(FileNotFoundError)
def missing_data(error):
    return jsonify(error=str(error)), 404


@simulator_bp.errorhandler(sf.LibsndfileError)
def bad_audio(error):
    return jsonify(error="샘플 WAV 파일을 읽을 수 없습니다."), 422


@lru_cache(maxsize=4)
def _cached_settings(path: str, modified_ns: int) -> dict:
    del modified_ns
    return load_settings(path)


def _resources():
    store = load_ais(current_app.config["AIS_CSV"])
    config_path = Path(current_app.config["SIMULATOR_CONFIG"]).expanduser().resolve()
    settings = _cached_settings(str(config_path), config_path.stat().st_mtime_ns)
    return store, config_path, settings


def _number(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(request.args.get(name, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: 숫자를 입력하세요.") from error
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}: {low}~{high} 범위로 입력하세요.")
    return value


def _query(store):
    bounds = store.bounds
    lat = _number("lat", (bounds[0][0] + bounds[1][0]) / 2, -90, 90)
    lon = _number("lon", (bounds[0][1] + bounds[1][1]) / 2, -180, 180)
    radius = _number("radius_km", 5, 0.5, 20)
    at = parse_utc(request.args.get("at", store.start_utc))
    if not store.start_epoch <= at.timestamp() <= store.end_epoch:
        raise ValueError("선택 시각이 AIS 기록 범위를 벗어났습니다.")
    return at, lat, lon, radius


def _window_minutes() -> int:
    window = _number("window_minutes", 60, 15, 120)
    if not window.is_integer():
        raise ValueError("window_minutes는 정수여야 합니다.")
    return int(window)


@simulator_bp.get("/meta")
def metadata():
    store, _config_path, settings = _resources()
    bounds = store.bounds
    return jsonify({
        **store.metadata(),
        "default_receiver": {"lat": (bounds[0][0] + bounds[1][0]) / 2,
                             "lon": (bounds[0][1] + bounds[1][1]) / 2},
        "default_radius_km": 5,
        "frequency_hz": settings["frequency_hz"],
        "absorption_db_per_km": round(thorp_db_per_km(float(settings["frequency_hz"])), 5),
        "audio_seconds": settings["audio"]["seconds"],
        "sample_dataset": settings["sample_dataset"],
        "assumptions": "선종별 소스 레벨은 예시 가정값입니다. 수심·수온·해저 지형은 반영하지 않습니다.",
    })


@simulator_bp.get("/scene")
def scene():
    store, _config_path, settings = _resources()
    at, lat, lon, radius = _query(store)
    window = _window_minutes()
    result = scene_at(store, at, lat, lon, radius, settings)
    result["series"] = level_series(store, at, lat, lon, radius, window, settings)
    result["frequency_hz"] = settings["frequency_hz"]
    return jsonify(result)


@simulator_bp.get("/audio")
def audio():
    store, config_path, settings = _resources()
    at, lat, lon, radius = _query(store)
    window = _window_minutes()
    result = scene_at(store, at, lat, lon, radius, settings, False)
    result["frequency_hz"] = settings["frequency_hz"]
    result["window_minutes"] = window
    result["sample_dataset"] = settings["sample_dataset"]
    result["series"] = level_series(store, at, lat, lon, radius, window,
                                    settings, include_vessels=True)
    content, used = mix_preview(result["vessels"], settings["sample_paths"],
                                current_app.config["AUDIO_ROOT"], settings)
    run_id = save_preview_run(current_app.config["SIMULATOR_RUNS_ROOT"], config_path,
                              result, used, content)
    response = send_file(io.BytesIO(content), mimetype="audio/wav",
                         download_name="simulated_hydrophone.wav")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Simulation-Run-Id"] = run_id
    return response

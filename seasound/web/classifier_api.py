# -*- coding: utf-8 -*-
"""web/classifier_api.py — Classifier 블루프린트 (WAV 업로드 -> 통합 클래스 판정)

app.py 에 두 줄 추가:
  from web.classifier_api import classifier_bp
  app.register_blueprint(classifier_bp)

API:
  GET  /api/classifier/model     현재 모델 정보 (어떤 best.pt 인지)
  POST /api/classifier/predict   form-data 'file' = WAV  -> 판정 JSON

모델 교체: configs/web.yaml 의 model_path 만 바꾸면, 다음 요청 때 자동으로 새 모델을 읽는다 (서버 재시작 불필요).
"""
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Blueprint, current_app, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
classifier_bp = Blueprint("classifier_api", __name__)

_state = {"cfg_mtime": None, "cfg": None, "clf": None, "model_path": None, "error": None}
_lock = threading.Lock()


def _cfg_path():
    return Path(current_app.config.get(
        "CLASSIFIER_CONFIG", os.environ.get("SEASOUND_CLASSIFIER_CONFIG", ROOT / "configs" / "web.yaml")))


def _get_classifier():
    """설정 파일이 바뀌었으면 다시 읽고, model_path 가 바뀌었으면 모델을 다시 로딩."""
    path = _cfg_path()
    with _lock:
        mtime = path.stat().st_mtime
        if mtime != _state["cfg_mtime"]:
            _state["cfg"] = yaml.safe_load(path.read_text(encoding="utf-8"))
            _state["cfg_mtime"] = mtime
        cfg = _state["cfg"]
        model_path = (ROOT / cfg["model_path"]).resolve()
        if model_path != _state["model_path"]:
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}. "
                                        f"{path.name} 의 model_path 를 확인하세요.")
            from ml.infer import AudioClassifier  # torch 로딩은 첫 요청 때만
            t0 = time.perf_counter()
            _state["clf"] = AudioClassifier(model_path, cfg.get("device", "auto"))
            _state["model_path"] = model_path
            _log(cfg, {"event": "model_loaded", **_state["clf"].info,
                       "load_sec": round(time.perf_counter() - t0, 2)})
        return _state["clf"], cfg


def _log(cfg, record):
    log_path = ROOT / cfg.get("log_path", "logs/inference.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "event": "inference", **record}
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@classifier_bp.get("/api/classifier/model")
def model_info():
    try:
        clf, _ = _get_classifier()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(clf.info)


@classifier_bp.post("/api/classifier/predict")
def predict():
    t0 = time.perf_counter()
    try:
        clf, cfg = _get_classifier()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    max_mb = int(cfg.get("max_upload_mb", 50))

    def fail(msg, code, **extra):
        _log(cfg, {"status": "rejected", "error": msg, **extra})
        return jsonify({"error": msg}), code

    up = request.files.get("file")
    if up is None or not up.filename:
        return fail("파일이 없습니다 (form 필드 이름: file)", 400)
    name = up.filename
    if not name.lower().endswith(".wav"):
        return fail("WAV 파일만 받습니다", 400, input_file=name)
    data = up.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        return fail(f"{max_mb}MB 이하만 받습니다", 413, input_file=name)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return fail("WAV 형식이 아닙니다", 400, input_file=name)
    try:
        res = clf.predict_bytes(data, min_duration_sec=float(cfg.get("min_duration_sec", 1.0)))
    except ValueError as e:
        return fail(str(e), 400, input_file=name)
    except Exception as e:
        return fail(f"읽을 수 없는 WAV입니다: {e}", 400, input_file=name)
    res["response_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["model"] = {"run_id": clf.info["run_id"], "n_fft": clf.info["n_fft"]}
    _log(cfg, {"status": "ok", "input_file": name, "input_bytes": len(data),
               "input_sha256": hashlib.sha256(data).hexdigest(),
               "model_run_id": clf.info["run_id"], "model_path": clf.info["model_path"],
               "label": res["label"], "probabilities": res["probabilities"],
               "duration_sec": res["duration_sec"], "n_segments": res["n_segments"],
               "padded": res["padded"], "gradcam_path": None,
               "inference_ms": res["elapsed_ms"], "response_ms": res["response_ms"]})
    return jsonify(res)
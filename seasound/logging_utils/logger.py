# -*- coding: utf-8 -*-
"""logging_utils/logger.py — 모든 단계가 같이 쓰는 JSON Lines 로그 도구."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _to_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    return str(obj)


class JsonlLogger:
    """log("event", key=value, ...) 를 호출하면 jsonl 파일에 한 줄씩 추가한다."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event, **fields):
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_to_json) + "\n")
        return record


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_to_json)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_run_id(tag, seed):
    return f"{datetime.now():%Y%m%d-%H%M%S}_{tag}_s{seed}"


def env_info():
    """실행 환경 기록 (명세: Python·OS·GPU 정보)."""
    info = {"python": sys.version.split()[0], "os": platform.platform(),
            "numpy": np.__version__}
    for name in ("scipy", "librosa", "torch", "torchvision", "sklearn"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return info

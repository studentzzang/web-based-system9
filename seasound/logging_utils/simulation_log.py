"""Save a replayable record of each generated listening preview."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def save_preview_run(runs_root: str | Path, config_path: str | Path,
                     scene: dict, used_samples: list[dict], audio_bytes: bytes) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"sim-{stamp}-{uuid.uuid4().hex[:8]}"
    destination = Path(runs_root) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, destination / "config.yaml")
    record = {
        "experiment_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "at_utc": scene["at_utc"],
        "receiver": scene["receiver"],
        "radius_km": scene["radius_km"],
        "vessel_count": scene["vessel_count"],
        "total_rl_db": scene["total_rl_db"],
        "frequency_hz": scene["frequency_hz"],
        "sample_dataset": scene["sample_dataset"],
        "window_minutes": scene["window_minutes"],
        "vessels": scene["vessels"],
        "series": scene["series"],
        "sample_sources": used_samples,
    }
    with (destination / "simulation.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (destination / "preview.wav").write_bytes(audio_bytes)
    return run_id

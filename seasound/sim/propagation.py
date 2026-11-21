"""Simplified, reproducible receiver-level calculations for historical AIS."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import yaml


EARTH_RADIUS_M = 6_371_000.0


def load_settings(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("시뮬레이션 설정 YAML이 비어 있거나 올바르지 않습니다.")
    for key in ("sample_dataset", "frequency_hz", "seed", "minimum_distance_m", "reference_speed_kn",
                "speed_db_per_doubling", "source_level_db", "sample_paths", "audio"):
        if key not in settings:
            raise ValueError(f"시뮬레이션 설정에 {key} 항목이 없습니다.")
    for key in ("frequency_hz", "minimum_distance_m", "reference_speed_kn"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"{key} 값은 양수여야 합니다.")
    if isinstance(settings["seed"], bool) or not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 합니다.")
    if not isinstance(settings["sample_dataset"], str) or not settings["sample_dataset"].strip():
        raise ValueError("sample_dataset 이름이 필요합니다.")
    if not isinstance(settings["source_level_db"], dict) or not isinstance(settings["sample_paths"], dict):
        raise ValueError("선종별 소음 레벨과 샘플 경로 설정이 필요합니다.")
    return settings


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def thorp_db_per_km(frequency_hz: float) -> float:
    """Thorp seawater absorption approximation (frequency in kHz)."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("주파수는 양수여야 합니다.")
    f2 = (frequency_hz / 1000.0) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 0.000275 * f2 + 0.003


def transmission_loss_db(distance_meters: float, frequency_hz: float,
                         minimum_distance_m: float = 100.0) -> float:
    if not math.isfinite(distance_meters) or distance_meters < 0:
        raise ValueError("거리는 0 이상의 유한한 값이어야 합니다.")
    distance = max(distance_meters, minimum_distance_m)
    return 20 * math.log10(distance) + thorp_db_per_km(frequency_hz) * distance / 1000.0


def source_level_db(label: str, sog_kn: float | None, settings: dict) -> float:
    levels = settings["source_level_db"]
    base = float(levels.get(label, levels.get("Other", 145.0)))
    speed = max(float(sog_kn or 0), 1.0)
    change = float(settings["speed_db_per_doubling"]) * math.log2(
        speed / float(settings["reference_speed_kn"])
    )
    return base + max(-12.0, min(12.0, change))


def combine_levels_db(levels: list[float]) -> float | None:
    if not levels:
        return None
    highest = max(levels)
    return highest + 10 * math.log10(sum(10 ** ((level - highest) / 10) for level in levels))


def scene_at(store, at: datetime, lat: float, lon: float, radius_km: float,
             settings: dict, include_positions: bool = True) -> dict:
    nearby = []
    positions = []
    for vessel in store.positions_at(at):
        meters = distance_m(lat, lon, vessel["lat"], vessel["lon"])
        inside = meters <= radius_km * 1000
        if include_positions:
            positions.append({key: vessel[key] for key in
                              ("mmsi", "vessel_name", "unified_label", "lat", "lon")})
            positions[-1]["inside_radius"] = inside
        if not inside:
            continue
        sl = source_level_db(vessel["unified_label"], vessel["sog_kn"], settings)
        tl = transmission_loss_db(meters, float(settings["frequency_hz"]),
                                  float(settings["minimum_distance_m"]))
        nearby.append({key: vessel[key] for key in
                       ("mmsi", "vessel_name", "unified_label", "lat", "lon",
                        "sog_kn", "timestamp_utc")})
        nearby[-1].update(distance_m=round(meters, 1), sl_db=round(sl, 2),
                          tl_db=round(tl, 2), rl_db=round(sl - tl, 2))
    nearby.sort(key=lambda item: item["rl_db"], reverse=True)
    return {
        "at_utc": at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "receiver": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "vessel_count": len(nearby),
        "total_rl_db": round(combine_levels_db([v["rl_db"] for v in nearby]), 2) if nearby else None,
        "vessels": nearby,
        "positions": positions,
    }


def level_series(store, at: datetime, lat: float, lon: float, radius_km: float,
                 window_minutes: int, settings: dict, include_vessels: bool = False) -> list[dict]:
    center = at.timestamp()
    duration = window_minutes * 60
    start = max(store.start_epoch, center - duration / 2)
    end = min(store.end_epoch, start + duration)
    start = max(store.start_epoch, end - duration)
    step = 5 * 60
    stamps = []
    current = start
    while current < end:
        stamps.append(current)
        current += step
    if not stamps or stamps[-1] != end:
        stamps.append(end)
    series = []
    for stamp in stamps:
        moment = datetime.fromtimestamp(stamp, timezone.utc)
        sample = scene_at(store, moment, lat, lon, radius_km, settings, False)
        frame = {"at_utc": sample["at_utc"], "total_rl_db": sample["total_rl_db"],
                 "vessel_count": sample["vessel_count"]}
        if include_vessels:
            frame["vessels"] = sample["vessels"]
        series.append(frame)
    return series

"""Read the cleaned AIS point file for the Traffic Map.

The file is intentionally outside Git, alongside the audio catalog. Loading is
cached by file modification time so timeline requests do not rescan the CSV.
"""

from __future__ import annotations

import csv
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


REQUIRED_COLUMNS = {
    "mmsi", "timestamp_utc", "lat", "lon", "sog_kn", "cog_deg",
    "vessel_type_code", "unified_label", "vessel_name",
}
CLASS_ORDER = (
    "Cargo", "Tanker", "Passenger", "Tug/Work", "Fishing", "Small craft", "Other",
)
MAX_POSITION_AGE_SECONDS = 15 * 60
TRAIL_SECONDS = 10 * 60


def parse_utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("UTC 시각 형식이 올바르지 않습니다.") from error
    if result.tzinfo is None:
        raise ValueError("UTC 시각에는 시간대를 지정해야 합니다.")
    return result.astimezone(timezone.utc)


def optional_float(value: str, minimum: float, maximum: float) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and minimum <= result <= maximum else None


@dataclass(frozen=True, slots=True)
class AISPoint:
    epoch: float
    timestamp_utc: str
    lat: float
    lon: float
    sog_kn: float | None
    cog_deg: float | None
    vessel_type_code: str
    label: str
    name: str


@dataclass(frozen=True, slots=True)
class VesselTrack:
    mmsi: str
    name: str
    label: str
    points: tuple[AISPoint, ...]
    epochs: tuple[float, ...]


class AISStore:
    def __init__(self, tracks: dict[str, VesselTrack], point_count: int):
        if not tracks:
            raise ValueError("표시할 AIS 위치 기록이 없습니다.")
        self.tracks = tracks
        self.point_count = point_count
        first = min(track.epochs[0] for track in tracks.values())
        last = max(track.epochs[-1] for track in tracks.values())
        self.start_utc = datetime.fromtimestamp(first, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.end_utc = datetime.fromtimestamp(last, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.start_epoch = first
        self.end_epoch = last
        self.bounds = [
            [min(point.lat for track in tracks.values() for point in track.points),
             min(point.lon for track in tracks.values() for point in track.points)],
            [max(point.lat for track in tracks.values() for point in track.points),
             max(point.lon for track in tracks.values() for point in track.points)],
        ]
        self.date_utc = self.start_utc[:10]
        class_counts = Counter(track.label for track in tracks.values())
        self.classes = [
            {"name": name, "vessels": class_counts[name]}
            for name in CLASS_ORDER if class_counts[name]
        ]

    def metadata(self) -> dict:
        return {
            "source": "NOAA Marine Cadastre AIS · 과거 위치 기록",
            "date_utc": self.date_utc,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "bounds": self.bounds,
            "point_count": self.point_count,
            "vessel_count": len(self.tracks),
            "classes": self.classes,
            "position_max_age_minutes": MAX_POSITION_AGE_SECONDS // 60,
        }

    def positions_at(self, at: datetime, label: str = "") -> list[dict]:
        epoch = at.timestamp()
        if not self.start_epoch <= epoch <= self.end_epoch:
            raise ValueError("선택한 시각이 AIS 데이터 범위를 벗어났습니다.")
        if label and label not in {item["name"] for item in self.classes}:
            raise ValueError("알 수 없는 선종 필터입니다.")

        visible = []
        for track in self.tracks.values():
            index = bisect_right(track.epochs, epoch) - 1
            if index < 0:
                continue
            point = track.points[index]
            age = epoch - point.epoch
            if age > MAX_POSITION_AGE_SECONDS or (label and point.label != label):
                continue
            trail_start = bisect_left(track.epochs, epoch - TRAIL_SECONDS, 0, index + 1)
            visible.append({
                "mmsi": track.mmsi,
                "vessel_name": point.name or track.name or f"MMSI {track.mmsi}",
                "unified_label": point.label,
                "vessel_type_code": point.vessel_type_code,
                "timestamp_utc": point.timestamp_utc,
                "age_seconds": int(age),
                "lat": point.lat,
                "lon": point.lon,
                "sog_kn": point.sog_kn,
                "cog_deg": point.cog_deg,
                "trail": [[item.lat, item.lon] for item in track.points[trail_start:index + 1]],
            })
        visible.sort(key=lambda item: (item["unified_label"], item["vessel_name"], item["mmsi"]))
        return visible

    def vessel_track(self, mmsi: str) -> dict:
        track = self.tracks.get(mmsi)
        if track is None:
            raise KeyError(mmsi)
        return {
            "mmsi": track.mmsi,
            "vessel_name": track.name or f"MMSI {mmsi}",
            "unified_label": track.label,
            "points": [
                {"timestamp_utc": point.timestamp_utc, "lat": point.lat, "lon": point.lon}
                for point in track.points
            ],
        }


@lru_cache(maxsize=2)
def _load_csv(path: str, modified_ns: int, size: int) -> AISStore:
    del modified_ns, size  # Their values form the cache key.
    grouped = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("AIS CSV 필수 열이 없습니다: " + ", ".join(sorted(missing)))
        for row in reader:
            mmsi = (row["mmsi"] or "").strip()
            if len(mmsi) != 9 or not mmsi.isdigit():
                continue
            try:
                timestamp = parse_utc(row["timestamp_utc"])
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            grouped[mmsi].append(AISPoint(
                epoch=timestamp.timestamp(),
                timestamp_utc=timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
                lat=lat,
                lon=lon,
                sog_kn=optional_float(row["sog_kn"], 0, 100),
                cog_deg=optional_float(row["cog_deg"], 0, 360),
                vessel_type_code=(row["vessel_type_code"] or "").strip(),
                label=(row["unified_label"] or "Other").strip(),
                name=(row["vessel_name"] or "").strip(),
            ))

    tracks = {}
    for mmsi, points in grouped.items():
        points.sort(key=lambda point: point.epoch)
        latest_named = next((point.name for point in reversed(points) if point.name), "")
        latest_labeled = next((point.label for point in reversed(points) if point.label != "Other"), points[-1].label)
        tracks[mmsi] = VesselTrack(
            mmsi=mmsi,
            name=latest_named,
            label=latest_labeled,
            points=tuple(points),
            epochs=tuple(point.epoch for point in points),
        )
    return AISStore(tracks, sum(len(track.points) for track in tracks.values()))


def load_ais(path: str | Path) -> AISStore:
    source = Path(path).expanduser().resolve()
    info = source.stat()
    return _load_csv(str(source), info.st_mtime_ns, info.st_size)

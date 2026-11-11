"""Validated, filesystem-safe recording catalog for SeaSound M1-01.

The catalog describes recordings; this module never opens or decodes WAV audio.
Callers supply both the catalog CSV path and the dataset root explicitly.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

REQUIRED_COLUMNS = (
    "recording_id",
    "dataset",
    "original_label",
    "common_label",
    "file_path",
    "duration_sec",
    "sample_rate_hz",
    "channels",
)
_RECORDING_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class CatalogError(ValueError):
    """Invalid catalog structure, metadata, or unsafe file reference."""


class RecordingNotFoundError(LookupError):
    """No catalog entry matches the requested recording ID."""


class RecordingFileError(FileNotFoundError):
    """A registered recording is unavailable on this computer."""


@dataclass(frozen=True)
class Recording:
    recording_id: str
    dataset: str
    original_label: str
    common_label: str | None
    file_path: str  # POSIX-style relative to Catalog.data_root, never an absolute path
    duration_sec: float
    sample_rate_hz: int
    channels: int


@dataclass(frozen=True)
class RecordingSummary:
    """Metadata safe to expose in a browser (intentionally excludes file_path)."""

    recording_id: str
    dataset: str
    original_label: str
    common_label: str | None
    duration_sec: float
    sample_rate_hz: int
    channels: int


@dataclass(frozen=True)
class Catalog:
    data_root: Path
    entries: Mapping[str, Recording]


@dataclass(frozen=True)
class RecordingLocation:
    recording: Recording
    absolute_path: Path  # for server-side use only; do not send to browsers


def _resolved_wav_path(data_root: Path, relative_path: str) -> Path:
    """Resolve a catalog path without allowing absolute paths, traversal, or symlink escape."""

    if not relative_path or "\\" in relative_path or ":" in relative_path or "\x00" in relative_path:
        raise CatalogError("file_path must be a POSIX-style relative WAV path")
    parts = relative_path.split("/")
    if relative_path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise CatalogError("file_path must not be absolute or contain '.', '..', or empty segments")
    if not relative_path.lower().endswith(".wav"):
        raise CatalogError("file_path must reference a .wav file")

    resolved = data_root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(data_root):
        raise CatalogError("file_path resolves outside the configured data_root")
    return resolved


def _positive_float(raw: str, column: str, row_number: int) -> float:
    try:
        number = float(raw)
    except ValueError as exc:
        raise CatalogError(f"row {row_number}: {column} must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise CatalogError(f"row {row_number}: {column} must be finite and greater than 0")
    return number


def _positive_int(raw: str, column: str, row_number: int) -> int:
    if not re.fullmatch(r"[0-9]+", raw):
        raise CatalogError(f"row {row_number}: {column} must be a positive integer")
    number = int(raw)
    if number <= 0:
        raise CatalogError(f"row {row_number}: {column} must be greater than 0")
    return number


def load_catalog(catalog_path: str | Path, data_root: str | Path) -> Catalog:
    """Load and validate a CSV catalog without reading audio files.

    data_root may be a user-managed directory outside the Git repository.
    Each CSV file_path is interpreted relative to that root.
    """

    root = Path(data_root).expanduser().resolve()
    entries: dict[str, Recording] = {}

    with Path(catalog_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise CatalogError("catalog CSV must contain a header row")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise CatalogError(f"missing catalog columns: {', '.join(sorted(missing))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise CatalogError("catalog contains duplicate column names")

        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise CatalogError(f"row {row_number}: wrong number of columns")
            row = {key: value.strip() for key, value in raw_row.items()}
            recording_id = row["recording_id"]
            if not _RECORDING_ID_PATTERN.fullmatch(recording_id):
                raise CatalogError(f"row {row_number}: invalid recording_id")
            if recording_id in entries:
                raise CatalogError(f"row {row_number}: duplicate recording_id '{recording_id}'")
            for column in ("dataset", "original_label", "file_path"):
                if not row[column]:
                    raise CatalogError(f"row {row_number}: {column} is required")

            _resolved_wav_path(root, row["file_path"])
            entries[recording_id] = Recording(
                recording_id=recording_id,
                dataset=row["dataset"],
                original_label=row["original_label"],
                common_label=row["common_label"] or None,
                file_path=row["file_path"],
                duration_sec=_positive_float(row["duration_sec"], "duration_sec", row_number),
                sample_rate_hz=_positive_int(row["sample_rate_hz"], "sample_rate_hz", row_number),
                channels=_positive_int(row["channels"], "channels", row_number),
            )

    return Catalog(data_root=root, entries=MappingProxyType(entries))


def list_recordings(
    catalog: Catalog,
    *,
    dataset: str | None = None,
    original_label: str | None = None,
    min_duration_sec: float | None = None,
) -> list[RecordingSummary]:
    """List public metadata, sorted predictably, without leaking local file paths."""

    if min_duration_sec is not None and (
        not math.isfinite(min_duration_sec) or min_duration_sec < 0
    ):
        raise ValueError("min_duration_sec must be finite and >= 0")

    matching = (
        rec
        for rec in catalog.entries.values()
        if (dataset is None or rec.dataset.casefold() == dataset.casefold())
        and (original_label is None or rec.original_label.casefold() == original_label.casefold())
        and (min_duration_sec is None or rec.duration_sec >= min_duration_sec)
    )
    return [
        RecordingSummary(
            recording_id=rec.recording_id,
            dataset=rec.dataset,
            original_label=rec.original_label,
            common_label=rec.common_label,
            duration_sec=rec.duration_sec,
            sample_rate_hz=rec.sample_rate_hz,
            channels=rec.channels,
        )
        for rec in sorted(
            matching, key=lambda rec: (rec.dataset.casefold(), rec.original_label.casefold(), rec.recording_id)
        )
    ]


def get_recording(catalog: Catalog, recording_id: str) -> RecordingLocation:
    """Look up a recording and check that its WAV exists inside the permitted data_root."""

    try:
        record = catalog.entries[recording_id]
    except KeyError as exc:
        raise RecordingNotFoundError(f"recording_id not found: {recording_id}") from exc

    # Re-resolve on access as a defense against changed symlinks after catalog loading.
    path = _resolved_wav_path(catalog.data_root, record.file_path)
    if not path.is_file():
        raise RecordingFileError(f"WAV for recording_id '{recording_id}' is unavailable")
    return RecordingLocation(recording=record, absolute_path=path)

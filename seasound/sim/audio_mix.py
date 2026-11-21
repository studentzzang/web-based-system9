"""Short, deterministic WAV previews from class-matched recordings."""

from __future__ import annotations

import hashlib
import io
import math
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def safe_sample_path(root: str | Path, relative: str) -> Path:
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.suffix.lower() != ".wav":
        raise ValueError("샘플 WAV 경로가 올바르지 않습니다.")
    base = Path(root).expanduser().resolve()
    target = base.joinpath(*path.parts).resolve()
    if not target.is_relative_to(base):
        raise ValueError("음원 폴더 밖의 파일은 사용할 수 없습니다.")
    if not target.is_file():
        raise FileNotFoundError(f"샘플 WAV를 찾지 못했습니다: {relative}")
    return target


@lru_cache(maxsize=32)
def _sample_segment(path_name: str, modified_ns: int, mmsi: str,
                    seconds: int, output_rate: int, seed: int) -> np.ndarray:
    del modified_ns  # Included in the cache key so edited WAVs are reloaded.
    with sf.SoundFile(path_name) as stream:
        source_frames = max(1, math.ceil(seconds * stream.samplerate))
        max_start = max(0, len(stream) - source_frames)
        digest = hashlib.sha256(f"{seed}:{mmsi}:{path_name}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % (max_start + 1)
        stream.seek(start)
        samples = stream.read(source_frames, dtype="float32", always_2d=True)
        source_rate = stream.samplerate
    if samples.size == 0:
        raise ValueError("샘플 WAV에 재생할 음향이 없습니다.")
    mono = samples.mean(axis=1)
    if source_rate != output_rate:
        factor = math.gcd(source_rate, output_rate)
        mono = resample_poly(mono, output_rate // factor, source_rate // factor)
    target_frames = seconds * output_rate
    if len(mono) < target_frames:
        mono = np.tile(mono, math.ceil(target_frames / len(mono)))
    return np.asarray(mono[:target_frames], dtype=np.float32)


def mix_preview(vessels: list[dict], samples_by_class: dict[str, str],
                audio_root: str | Path, settings: dict) -> tuple[bytes, list[dict]]:
    options = settings["audio"]
    seconds = int(options["seconds"])
    rate = int(options["sample_rate_hz"])
    if not 1 <= seconds <= 20 or not 8000 <= rate <= 48000:
        raise ValueError("합성 음향 설정 범위가 올바르지 않습니다.")
    mix = np.zeros(seconds * rate, dtype=np.float32)
    used = []
    for vessel in vessels:
        label = vessel["unified_label"]
        relative = samples_by_class.get(label) or samples_by_class.get("Other")
        if not relative:
            continue
        path = safe_sample_path(audio_root, relative)
        segment = _sample_segment(str(path), path.stat().st_mtime_ns,
                                  str(vessel["mmsi"]), seconds, rate, int(settings["seed"]))
        centered = segment - float(np.mean(segment))
        rms = float(np.sqrt(np.mean(centered ** 2)))
        if rms < 1e-6:
            continue
        target_rms = float(options["rms_at_reference"]) * 10 ** (
            (float(vessel["rl_db"]) - float(options["reference_rl_db"])) / 20
        )
        target_rms = min(0.5, max(0.0, target_rms))
        mix += centered * (target_rms / rms)
        used.append({"mmsi": vessel["mmsi"], "unified_label": label,
                     "sample_path": relative, "rl_db": vessel["rl_db"]})
    fade = min(rate // 20, len(mix) // 2)
    if fade:
        envelope = np.linspace(0, 1, fade, endpoint=False, dtype=np.float32)
        mix[:fade] *= envelope
        mix[-fade:] *= envelope[::-1]
    # Fixed gain mapping preserves the effect of distance; only peaks are limited.
    mix = np.tanh(mix)
    output = io.BytesIO()
    sf.write(output, mix, rate, format="WAV", subtype="PCM_16")
    return output.getvalue(), used

"""WAV metadata and PCM loading for SeaSound M1-02.

This module reads audio only; it does not resample, downmix, or run STFT.
Use ``load_audio_segment`` to avoid reading an entire multi-minute WAV when
only one five-second Explorer segment is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioReadError(OSError):
    """File is missing, unreadable, corrupt, or not in WAV format."""


class AudioValidationError(ValueError):
    """Invalid numeric input or an audio signal that cannot be processed."""


class SegmentOutOfRangeError(AudioValidationError):
    """Requested interval contains fewer samples than the full requested duration."""


@dataclass(frozen=True)
class AudioInfo:
    sample_rate_hz: int
    channels: int
    frames: int
    subtype: str

    @property
    def duration_sec(self) -> float:
        return self.frames / self.sample_rate_hz


@dataclass(frozen=True)
class RawAudio:
    """Unmodified (except decoding to float32) WAV samples, shape (frames, channels)."""

    samples: np.ndarray
    sample_rate_hz: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate_hz


def _open_wav(path: str | Path) -> sf.SoundFile:
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise AudioReadError("Only .wav files are supported")
    try:
        audio_file = sf.SoundFile(str(path), mode="r")
    except (OSError, RuntimeError) as exc:
        raise AudioReadError("WAV file cannot be opened") from exc
    if audio_file.format != "WAV":
        audio_file.close()
        raise AudioReadError("The file extension is .wav but the content is not WAV")
    if audio_file.samplerate <= 0 or audio_file.channels <= 0:
        audio_file.close()
        raise AudioReadError("WAV has invalid sample rate or channel count")
    return audio_file


def inspect_wav(path: str | Path) -> AudioInfo:
    """Read the WAV header without decoding the whole recording."""

    with _open_wav(path) as audio_file:
        return AudioInfo(
            sample_rate_hz=audio_file.samplerate,
            channels=audio_file.channels,
            frames=audio_file.frames,
            subtype=audio_file.subtype,
        )


def load_audio(path: str | Path) -> RawAudio:
    """Decode the full WAV into float32; preserve all original channels."""

    with _open_wav(path) as audio_file:
        try:
            samples = audio_file.read(dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise AudioReadError("WAV samples could not be decoded") from exc
        rate = audio_file.samplerate
    if samples.shape[0] == 0:
        raise AudioReadError("WAV contains no audio frames")
    return RawAudio(samples=samples, sample_rate_hz=rate)


def load_audio_segment(
    path: str | Path, *, start_sec: float = 0.0, duration_sec: float = 5.0
) -> RawAudio:
    """Read one complete interval at the *original* sample rate.

    The beginning is rounded to the nearest source sample. We read exactly
    round(duration_sec * sample_rate) source frames, without zero padding or
    shortening the requested interval. A segment exceeding EOF is rejected.
    """

    if not math.isfinite(start_sec) or start_sec < 0:
        raise AudioValidationError("start_sec must be finite and >= 0")
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise AudioValidationError("duration_sec must be finite and > 0")

    with _open_wav(path) as audio_file:
        rate = audio_file.samplerate
        start_frame = int(math.floor(start_sec * rate + 0.5))
        window_frames = int(math.floor(duration_sec * rate + 0.5))
        if window_frames <= 0:
            raise AudioValidationError("duration_sec is shorter than one source sample")
        if start_frame + window_frames > audio_file.frames:
            raise SegmentOutOfRangeError("The requested interval exceeds the WAV duration")
        audio_file.seek(start_frame)
        try:
            samples = audio_file.read(frames=window_frames, dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise AudioReadError("WAV segment could not be decoded") from exc

    if samples.shape[0] != window_frames:
        raise AudioReadError("WAV did not contain all requested frames")
    return RawAudio(samples=samples, sample_rate_hz=rate)

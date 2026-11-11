"""Mono conversion, resampling and exact-duration segment extraction (M1-02)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from data_io.audio import AudioValidationError, RawAudio, SegmentOutOfRangeError


@dataclass(frozen=True)
class PreparedAudio:
    """One-channel float32 samples at the configured target sample rate."""

    samples: np.ndarray  # 1-dimensional, float32
    sample_rate_hz: int

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate_hz


def prepare_audio(raw: RawAudio, *, target_sample_rate_hz: int = 16_000) -> PreparedAudio:
    """Downmix all channels by arithmetic mean and polyphase-resample to target.

    Input amplitude is not peak/volume normalized: downstream STFT determines
    its own relative-dB reference. The raw samples remain unchanged.
    """

    if isinstance(target_sample_rate_hz, bool) or not isinstance(target_sample_rate_hz, int) or target_sample_rate_hz <= 0:
        raise AudioValidationError("target_sample_rate_hz must be a positive integer")
    if not isinstance(raw.sample_rate_hz, int) or raw.sample_rate_hz <= 0:
        raise AudioValidationError("raw.sample_rate_hz must be a positive integer")
    samples = raw.samples
    if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise AudioValidationError("raw audio samples must have shape (frames, channels), both > 0")
    if not np.isfinite(samples).all():
        raise AudioValidationError("raw audio contains NaN or infinity")

    # Float64 accumulation avoids rounding issues when combining many channels.
    mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    if raw.sample_rate_hz != target_sample_rate_hz:
        divisor = math.gcd(raw.sample_rate_hz, target_sample_rate_hz)
        mono = resample_poly(
            mono,
            up=target_sample_rate_hz // divisor,
            down=raw.sample_rate_hz // divisor,
        ).astype(np.float32, copy=False)
    if not np.isfinite(mono).all():
        raise AudioValidationError("resampling produced non-finite audio")
    return PreparedAudio(samples=mono, sample_rate_hz=target_sample_rate_hz)


def extract_segment(
    prepared: PreparedAudio, *, start_sec: float = 0.0, duration_sec: float = 5.0
) -> PreparedAudio:
    """Take a complete segment; never pad or silently truncate it.

    This is useful after loading and preparing a *full recording*. For M1
    Explorer's memory-efficient path, prefer load_audio_segment() followed by
    prepare_audio(), where the start_sec is applied at the source sample rate.
    """

    if not math.isfinite(start_sec) or start_sec < 0:
        raise AudioValidationError("start_sec must be finite and >= 0")
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise AudioValidationError("duration_sec must be finite and > 0")
    if not isinstance(prepared.sample_rate_hz, int) or prepared.sample_rate_hz <= 0:
        raise AudioValidationError("sample_rate_hz must be a positive integer")
    if not isinstance(prepared.samples, np.ndarray) or prepared.samples.ndim != 1:
        raise AudioValidationError("prepared samples must be a 1D array")
    rate = prepared.sample_rate_hz
    start = int(math.floor(start_sec * rate + 0.5))
    size = int(math.floor(duration_sec * rate + 0.5))
    if size <= 0:
        raise AudioValidationError("duration_sec is shorter than one target sample")
    if start + size > prepared.samples.size:
        raise SegmentOutOfRangeError("The requested interval exceeds the available audio duration")
    return PreparedAudio(samples=prepared.samples[start:start + size].copy(), sample_rate_hz=rate)

"""LOFAR tonal detection with local background and frame continuity."""
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter1d

from dsp.explorer_spectrogram import lofar_chunk


TONAL_BAND_HZ = (10, 500)
FRAME_SECONDS = 0.25
CONTEXT_SECONDS = 3
MIN_PEAK_DB = 6.0
MIN_STANDARD_SCORE = 2.5
MIN_TEMPORAL_SUPPORT = 0.5

# Frequencies one or two bins from the center can contain Hann-window leakage.
NOISE_WINDOW = np.ones(25, dtype=np.float64)
NOISE_WINDOW[10:15] = 0
NOISE_WINDOW /= NOISE_WINDOW.sum()

# Eight frames on either side, excluding the candidate frame itself.
TIME_WINDOW = np.ones(17, dtype=np.float64)
TIME_WINDOW[8] = 0


def sustained_excess_power(levels):
    """Return linear power above the local background at supported peaks."""
    levels = levels.astype(np.float64, copy=False)
    first_floor = convolve1d(levels, NOISE_WINDOW, axis=0, mode="nearest")
    clipped = np.where(levels > first_floor + MIN_PEAK_DB, first_floor, levels)
    floor = convolve1d(clipped, NOISE_WINDOW, axis=0, mode="nearest")
    mean_square = convolve1d(clipped ** 2, NOISE_WINDOW, axis=0, mode="nearest")
    deviation = np.sqrt(np.maximum(mean_square - floor ** 2, 1.0))
    above_floor = levels - floor
    peaks = levels == maximum_filter1d(levels, size=3, axis=0, mode="nearest")
    candidates = peaks & (above_floor >= MIN_PEAK_DB)
    candidates &= above_floor / deviation >= MIN_STANDARD_SCORE
    candidates &= levels >= levels.max(axis=0) - 50

    nearby = maximum_filter1d(candidates, size=3, axis=0, mode="constant")
    support = convolve1d(nearby.astype(np.float64), TIME_WINDOW, axis=1,
                         mode="constant")
    available = convolve1d(np.ones(levels.shape[1]), TIME_WINDOW, mode="constant")
    supported = candidates & (available > 0) & (support >= MIN_TEMPORAL_SUPPORT * available)
    excess = 10 ** (levels / 10) - 10 ** (floor / 10)
    return np.where(supported, np.maximum(excess, 0), 0)


def representative_tonal(source, windows):
    """Rank one-bin groups by time-integrated, linear excess power."""
    rate = source.samplerate
    total_power = hit_counts = lower_frames = upper_frames = frequencies = None
    for first, last in windows:
        start = max(0, first - CONTEXT_SECONDS * rate)
        end = min(source.frames, last + CONTEXT_SECONDS * rate)
        source.seek(start)
        samples = source.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
        f, times, levels = lofar_chunk(samples, rate)
        times += start / rate
        # Context supplies neighboring frames, while only the actual window votes.
        recorded = times < end / rate
        times = times[recorded]
        powers = sustained_excess_power(levels[:, recorded])
        own = (times >= first / rate) & (times < last / rate)
        band = (f >= TONAL_BAND_HZ[0]) & (f <= TONAL_BAND_HZ[1])
        f, powers = f[band], powers[band][:, own]
        if total_power is None:
            frequencies = f
            total_power = np.zeros(len(f) - 1)
            hit_counts = np.zeros(len(f) - 1, dtype=int)
            lower_frames = np.zeros_like(hit_counts)
            upper_frames = np.zeros_like(hit_counts)

        lower, upper = powers[:-1], powers[1:]
        # Keep one observation per frame for a group spanning adjacent bins.
        combined = np.maximum(lower, upper)
        frame_seconds = np.minimum(FRAME_SECONDS, last / rate - times[own])
        # A centered 1-second window can leak into the neighboring segment.
        observed = np.count_nonzero(combined, axis=1) >= 2
        total_power += np.where(observed, combined @ frame_seconds, 0)
        hit_counts += observed
        lower_frames += observed * np.count_nonzero(
            (lower >= upper) & (lower > 0), axis=1)
        upper_frames += observed * np.count_nonzero(upper > lower, axis=1)

    if total_power is None or not np.any(total_power > 0):
        return []

    index = max(range(len(total_power)), key=lambda i: (total_power[i], hit_counts[i]))
    hits = [lower_frames[index], upper_frames[index]]
    hz = float(np.median(np.repeat(frequencies[index:index + 2], hits)))
    return [{"hz": round(hz, 3), "segments": int(hit_counts[index])}]

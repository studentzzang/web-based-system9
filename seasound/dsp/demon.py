"""Multiband DEMON spectra and conditional shaft-rate evidence."""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, periodogram, sosfiltfilt

from data_io.preprocess import resample_audio


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/analysis.yaml"


def read_settings():
    try:
        with CONFIG_PATH.open(encoding="utf8") as stream:
            cfg = yaml.safe_load(stream)["demon"]
        bands = np.asarray(cfg["carrier_bands_hz"], dtype=float)
        low, high = cfg["modulation_range_hz"]
        bounds_ok = (bands.ndim == 2 and len(bands) > 0 and bands.shape[1] == 2
                     and np.isfinite(bands).all() and (bands[:, 0] >= 0).all()
                     and (bands[:, 1] > bands[:, 0]).all()
                     and (bands[1:, 0] >= bands[:-1, 1]).all())
        ratios = [cfg[key] for key in ("min_peak_power_fraction", "min_explained_power",
                  "min_window_fraction", "min_score", "min_score_margin")]
        weights = cfg["score_weights"]
        values = [weights[k] for k in ("coverage", "explained_power", "harmonic_count")]
        if not (all(0 <= w <= 1 for w in values) and math.isclose(sum(values), 1)
                and type(cfg["min_rpm_orders"]) is int and 3 <= cfg["min_rpm_orders"] <= cfg["max_harmonic"]
                and bounds_ok and isinstance(cfg["envelope_rate"], int)
                and 0 < low < high < cfg["envelope_rate"] / 2
                and isinstance(cfg["max_harmonic"], int) and 3 <= cfg["max_harmonic"] <= 32
                and all(0 < value <= 1 for value in ratios)
                and all(math.isfinite(cfg[k]) and cfg[k] > 0 for k in
                        ("local_noise_width_hz", "min_peak_db", "match_bins"))):
            raise ValueError
        cfg["carrier_bands_hz"] = bands.tolist()
        return cfg
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("configs/analysis.yaml의 demon 설정을 확인하세요.") from error


def envelope(samples, rate, low, high):
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    if low > 0 and high < rate / 2:
        sos = butter(4, (low, high), btype="bandpass", fs=rate, output="sos")
    elif low > 0:
        sos = butter(4, low, btype="highpass", fs=rate, output="sos")
    elif high < rate / 2:
        sos = butter(4, high, btype="lowpass", fs=rate, output="sos")
    else:
        sos = None
    if sos is not None:
        x = sosfiltfilt(sos, x)
    env = np.abs(hilbert(x))
    trim = min(rate // 4, len(env) // 8)
    return env[trim:-trim] if trim else env


def spectrum(samples, rate, cfg, bounds=None, min_fft_seconds=0):
    """Sum linear envelope power across passbands. A quiet band gets no extra weight."""
    bands = [bounds] if bounds is not None else cfg["carrier_bands_hz"]
    env_rate = cfg["envelope_rate"]
    combined = None
    if not np.isfinite(samples).all():
        raise ValueError("WAV에 유효하지 않은 신호 값이 있습니다.")
    for low, high in bands:
        high = min(high, rate / 2)
        if low >= high:
            continue
        env = envelope(samples, rate, low, high)
        env, _ = resample_audio(env, rate, target_rate=env_rate)
        nfft = max(len(env), min_fft_seconds * env_rate)
        freq, power = periodogram(env, fs=env_rate, window="hann", nfft=nfft,
                                  detrend="constant", scaling="density")
        if env.std() < max(1e-12, .02 * env.mean()):
            power.fill(0)
        if combined is None:
            combined = power.astype(np.float64)
        else:
            combined += power
    if combined is None:
        raise ValueError("샘플레이트에 맞는 DEMON 입력 대역이 없습니다.")
    return freq, combined, env_rate / len(env)


@dataclass(frozen=True)
class Evidence:
    hz: float
    orders: tuple
    share: float


def analyze_window(samples, rate, cfg, bounds=None):
    freq, power, resolution = spectrum(samples, rate, cfg, bounds)
    low, high = cfg["modulation_range_hz"]
    mask = (freq >= low) & (freq <= high)
    total = power[mask].sum()
    if not total > 0:
        return None, resolution, []
    step = freq[1]
    levels = 10 * np.log10(np.maximum(power, 1e-30))
    width = max(3, round(cfg["local_noise_width_hz"] / step) | 1)
    floor = median_filter(levels, size=min(width, 2 * len(freq) - 1), mode="nearest")
    peaks, _ = find_peaks(levels, prominence=cfg["min_peak_db"])
    peaks = peaks[mask[peaks] & (levels[peaks]-floor[peaks] >= cfg["min_peak_db"])
                  & (power[peaks] >= cfg["min_peak_power_fraction"] * total)]
    if len(peaks) == 0:
        return None, resolution, []
    # Refine the Hann-window peaks and integrate the power around each peak.
    curvature = levels[peaks-1] - 2*levels[peaks] + levels[peaks+1]
    offset = np.divide(levels[peaks-1] - levels[peaks+1], 2*curvature,
                       out=np.zeros_like(curvature), where=curvature != 0)
    lines = freq[peaks] + step * np.clip(offset, -.5, .5)
    dominant = float(lines[np.argmax(power[peaks])])
    if len(lines) < 2:
        return dominant, resolution, []
    orders = np.arange(1, cfg["max_harmonic"] + 1)
    seeds = np.unique(np.r_[np.diff(lines), (lines[:, None] / orders).ravel()])
    tol = cfg["match_bins"] * resolution
    proposals = []
    for seed in seeds[(seeds >= max(low, 2*tol)) & (seeds <= high/2 + tol)]:
        multiples = np.rint(lines / seed).astype(int)
        matched = ((multiples >= 1) & (multiples <= orders[-1])
                   & (np.abs(lines - multiples * seed) <= tol))
        indices = np.flatnonzero(matched)
        if len(indices) < 2:
            continue
        # Each harmonic order can contribute at most one measured line.
        ids = sorted(indices, key=lambda i: abs(lines[i] - multiples[i]*seed))
        selected = {}
        for i in ids:
            selected.setdefault(int(multiples[i]), i)
        harmonics = tuple(sorted(selected))
        if len(harmonics) < 2 or math.gcd(*harmonics) != 1:
            continue
        ids = np.array(list(selected.values()))
        bins = np.unique(peaks[ids, None] + np.array([-1, 0, 1]))
        share = float(power[bins[mask[bins]]].sum() / total)
        if share < cfg["min_explained_power"]:
            continue
        hz = float(np.median(lines[ids] / multiples[ids]))
        proposals.append(Evidence(hz, harmonics, share))
    # Nearby seed values describe the same observed group.
    unique = []
    for item in sorted(proposals, key=lambda e: (len(e.orders), e.share), reverse=True):
        if all(abs(item.hz - kept.hz) > tol for kept in unique):
            unique.append(item)
    return dominant, resolution, unique


def rank_windows(windows, cfg):
    """Score coverage, explained modulation power and harmonic count (not probability)."""
    points = sorted((item.hz, index, res, item)
                    for index, (_, res, items) in enumerate(windows) for item in items)
    ranked = []
    start = 0
    while start < len(points):
        hz, _, first_res, _ = points[start]
        end = start + 1
        while end < len(points):
            other, _, res, _ = points[end]
            if other-hz > cfg["match_bins"] * min(first_res, res):
                break
            end += 1
        best = {}
        for _, index, _, item in points[start:end]:
            if index not in best or item.share > best[index].share:
                best[index] = item
        start = end
        items = list(best.values())
        fraction = len(items) / len(windows)
        orders = sorted({order for item in items for order in item.orders})
        share = float(np.mean([item.share for item in items]))
        weights = cfg["score_weights"]
        score = (weights["coverage"] * fraction + weights["explained_power"] * share
                 + weights["harmonic_count"] * min(len(orders)/4, 1))
        ranked.append(dict(hz=round(float(np.median([e.hz for e in items])), 3),
                           segments=len(items), coverage=fraction, orders=orders,
                           explained_power=share, score=score))
    ranked.sort(key=lambda x: (-x["score"], -x["segments"], x["hz"]))
    accepted = []
    if ranked:
        best = ranked[0]
        runner_up = ranked[1]["score"] if len(ranked) > 1 else 0
        if (best["coverage"] >= cfg["min_window_fraction"] and best["score"] >= cfg["min_score"]
                and best["score"]-runner_up >= cfg["min_score_margin"]
                and 1 in best["orders"] and len(best["orders"]) >= cfg["min_rpm_orders"]):
            accepted = [dict(best, rpm=round(best["hz"] * 60, 2))]
    return ranked[:3], accepted


def blade_pass_rpm(windows, blades, cfg):
    """Conditional conversion only when one dominant line recurs sufficiently often."""
    peaks = sorted((hz, res) for hz, res, _ in windows if hz is not None)
    groups = []
    for hz, res in peaks:
        if not groups or hz-groups[-1][0][0] > cfg["match_bins"] * min(res, groups[-1][0][1]):
            groups.append([])
        groups[-1].append((hz, res))
    groups.sort(key=len, reverse=True)
    if not groups or (len(groups) > 1 and len(groups[0]) == len(groups[1])):
        return []
    group = groups[0]
    coverage = len(group) / len(windows)
    if coverage < cfg["min_window_fraction"]:
        return []
    blade_hz = float(np.median([p[0] for p in group]))
    hz = blade_hz / blades
    return [dict(hz=round(hz, 3), rpm=round(hz * 60, 2), segments=len(group),
                 coverage=coverage, blade_count=blades, blade_pass_hz=round(blade_hz, 3))]

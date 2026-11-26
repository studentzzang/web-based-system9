"""Multiband DEMON spectra and conditional shaft-rate evidence."""
import math
from dataclasses import dataclass, replace
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
        cfg.setdefault("band_analysis", "combined")
        cfg.setdefault("min_band_power_fraction", 0.0001)
        if (cfg["band_analysis"] not in ("combined", "independent")
                or not 0 <= cfg["min_band_power_fraction"] <= 1):
            raise ValueError
        cfg.setdefault("min_contiguous_windows", 4)
        cfg.setdefault("window_seconds", 20)
        cfg.setdefault("overlap", 0.0)
        cfg.setdefault("tracking", True)
        cfg.setdefault("max_drift_hz_per_second", 0.008)
        cfg.setdefault("max_gap_windows", 1)
        cfg.setdefault("min_shared_orders", 2)
        if not (type(cfg["window_seconds"]) is int and 5 <= cfg["window_seconds"] <= 60
                and 0 <= cfg["overlap"] <= 0.75
                and type(cfg["min_contiguous_windows"]) is int and cfg["min_contiguous_windows"] >= 2
                and type(cfg["tracking"]) is bool
                and 0 < cfg["max_drift_hz_per_second"] <= 0.1
                and type(cfg["max_gap_windows"]) is int and 0 <= cfg["max_gap_windows"] <= 3
                and type(cfg["min_shared_orders"]) is int and 1 <= cfg["min_shared_orders"] <= 8):
            raise ValueError
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
    # Observed orders and their relative peak powers, not blade identities.
    order_powers: tuple = ()
    bands: tuple = ()


def analyze_spectrum(freq, power, resolution, cfg):
    """Detect modulation and harmonic families in one power spectrum."""
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
        order_powers = tuple((int(multiples[i]), float(power[peaks[i]] / total)) for i in ids)
        proposals.append(Evidence(hz, harmonics, share, order_powers))
    # Nearby seed values describe the same observed group.
    unique = []
    for item in sorted(proposals, key=lambda e: (len(e.orders), e.share), reverse=True):
        if all(abs(item.hz - kept.hz) > tol for kept in unique):
            unique.append(item)
    return dominant, resolution, unique


def analyze_window(samples, rate, cfg, bounds=None):
    """Keep the combined dominant line, but detect harmonic evidence per band."""
    if bounds is not None or cfg.get("band_analysis", "combined") == "combined":
        return analyze_spectrum(*spectrum(samples, rate, cfg, bounds), cfg)

    bands = []
    for low, high in cfg["carrier_bands_hz"]:
        high = min(high, rate / 2)
        if low < high:
            bands.append(((low, high), spectrum(samples, rate, cfg, (low, high))))
    if not bands:
        raise ValueError("샘플레이트에 맞는 DEMON 입력 대역이 없습니다.")

    freq, _, resolution = bands[0][1]
    combined = np.sum([power for _, (_, power, _) in bands], axis=0)
    dominant, _, _ = analyze_spectrum(freq, combined, resolution, cfg)
    lo, hi = cfg["modulation_range_hz"]
    mask = (freq >= lo) & (freq <= hi)
    total = float(combined[mask].sum())
    if total <= 0:
        return dominant, resolution, []

    # Compare candidates, not raw band power. Exclude negligible modulation
    # energy to avoid amplifying filter leakage or numerical noise.
    candidates = []
    for band, (f, power, res) in bands:
        if float(power[mask].sum()) < cfg.get("min_band_power_fraction", 0.0001) * total:
            continue
        _, _, items = analyze_spectrum(f, power, res, cfg)
        candidates.extend(replace(item, bands=(band,)) for item in items)

    # Keep one actual observed family for each nearby base frequency. Do not
    # invent extra orders by combining unrelated peaks from different bands.
    weights = cfg["score_weights"]
    def quality(item):
        return (weights["explained_power"] * item.share
                + weights["harmonic_count"] * min(len(item.orders) / 4, 1))

    unique = []
    tol = cfg["match_bins"] * resolution
    for item in sorted(candidates, key=lambda e: (-quality(e), e.hz, e.orders)):
        # Preserve the best actually observed family; record supporting bands.
        match = next((i for i, kept in enumerate(unique)
                      if abs(item.hz - kept.hz) <= tol
                      and len(set(item.orders) & set(kept.orders)) >= 2), None)
        if match is None:
            unique.append(item)
        else:
            kept = unique[match]
            unique[match] = replace(kept, bands=tuple(sorted(set(kept.bands + item.bands))))
    return dominant, resolution, unique


def rank_fixed_windows(windows, cfg, centers, limit):
    """Score coverage, explained modulation power and harmonic count (not probability)."""
    points = sorted(((item.hz, index, res, item)
                     for index, (_, res, items) in enumerate(windows) for item in items),
                    key=lambda point: point[:3])
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
        qualified = sum(_qualified(item, cfg) for item in items)
        ranked.append(dict(hz=round(float(np.median([e.hz for e in items])), 3),
                           qualified_segments=qualified, qualified_coverage=qualified / len(windows),
                           track=[dict(window=i, time_seconds=round(centers[i], 3),
                                       hz=round(e.hz, 3), orders=list(e.orders))
                                  for i, e in sorted(best.items())],
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
    return ranked[:limit], accepted


def window_ranges(frames, rate, cfg):
    """Bounded, chronological windows; last window is aligned to EOF."""
    if frames < 2 * rate:
        return []
    width = int(cfg.get("window_seconds", 20) * rate)
    hop = max(1, round(width * (1 - cfg.get("overlap", 0.0))))
    if hop == width:
        ranges = []
        for start in range(0, frames, width):
            if frames - start < 2 * rate and ranges:
                ranges[-1] = (ranges[-1][0], frames)
                break
            ranges.append((start, min(start + width, frames)))
        return ranges
    if frames <= width:
        return [(0, frames)]
    starts = list(range(0, frames - width + 1, hop))
    if starts[-1] != frames - width:
        starts.append(frames - width)
    return [(start, start + width) for start in starts]


def _quality(item, cfg):
    w = cfg["score_weights"]
    return w["explained_power"] * item.share + w["harmonic_count"] * min(len(item.orders)/4, 1)


def _qualified(item, cfg):
    # The required orders must coexist in a window, not just in a clip-wide union.
    return 1 in item.orders and len(item.orders) >= cfg["min_rpm_orders"]


def rank_windows(windows, cfg, centers=None, *, limit=3):
    """Associate nearby harmonic families in time, then score entire tracks.

    Coverage counts overlapping analysis windows, not independent observations.
    A periodic acoustic source is a shaft hypothesis, never a measured shaft RPM.
    """
    if not windows:
        return [], []
    hop = cfg.get("window_seconds", 20) * (1 - cfg.get("overlap", 0.5))
    centers = list(centers) if centers is not None else [i*hop for i in range(len(windows))]
    if len(centers) != len(windows) or any(b <= a for a, b in zip(centers, centers[1:])):
        raise ValueError("DEMON window centers must be strictly increasing")
    if not cfg.get("tracking", True):
        return rank_fixed_windows(windows, cfg, centers, limit)
    # Find globally supported paths rather than greedily following the nearest
    # line: overlapping windows can contain both old and new rotational rates.
    nodes = [(i, res, item) for i, (_, res, items) in enumerate(windows) for item in items]
    predecessors = [[] for _ in nodes]
    by_window = {}
    for ni, (i, res, item) in enumerate(nodes):
        for prev_i in range(max(0, i-cfg.get("max_gap_windows", 1)-1), i):
            dt = centers[i] - centers[prev_i]
            for pi in by_window.get(prev_i, []):
                _, prev_res, prev = nodes[pi]
                tol = max(res, prev_res)*cfg["match_bins"] + dt*cfg.get("max_drift_hz_per_second", .008)
                distance = abs(item.hz-prev.hz)
                shared = len(set(item.orders) & set(prev.orders))
                if distance <= tol and shared >= cfg.get("min_shared_orders", 2):
                    predecessors[ni].append((pi, .08*distance/tol + .03*(i-prev_i-1)))
        by_window.setdefault(i, []).append(ni)
    rewards = [0.5 + _quality(item, cfg) + .10*_qualified(item, cfg) for _, _, item in nodes]
    tracks, remaining = [], set(range(len(nodes)))
    # Retain up to twelve competing paths; no reuse of a detected family.
    while remaining and len(tracks) < 12:
        best, parents = {}, {}
        for ni in sorted(remaining):
            best[ni], parents[ni] = rewards[ni], None
            for pi, penalty in predecessors[ni]:
                if pi in best and best[pi]+rewards[ni]-penalty > best[ni]:
                    best[ni], parents[ni] = best[pi]+rewards[ni]-penalty, pi
        end = max(best, key=best.get)
        path = []
        while end is not None:
            path.append(end)
            end = parents[end]
        path.reverse()
        tracks.append([nodes[ni] for ni in path])
        remaining.difference_update(path)

    ranked = []
    weights = cfg["score_weights"]
    for track in tracks:
        items = [entry[2] for entry in track]
        qualified = [item for item in items if _qualified(item, cfg)]
        coverage = len(items)/len(windows)
        qualified_coverage = len(qualified)/len(windows)
        share = float(np.mean([item.share for item in items]))
        harmonic_score = float(np.mean([min(len(item.orders)/4, 1) for item in items]))
        score = weights["coverage"]*coverage + weights["explained_power"]*share + weights["harmonic_count"]*harmonic_score
        # Blade-pass hypotheses: a repeatedly dominant observed harmonic >=2.
        # No automatic claim that this harmonic really is blade passage.
        votes = {}
        for item in qualified:
            if not item.order_powers:
                continue
            order = max(item.order_powers, key=lambda pair: pair[1])[0]
            if order >= 2:
                votes[order] = votes.get(order, 0) + 1
        hz = float(np.median([item.hz for item in items]))
        bpf = [dict(blade_count=n, hz=round(n*hz, 3), support=count/len(windows))
               for n, count in sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))
               if count/len(windows) >= cfg["min_window_fraction"]]
        values = [item.hz for item in items]
        ranked.append(dict(hz=round(hz, 3), segments=len(items), coverage=coverage,
            qualified_coverage=qualified_coverage, qualified_segments=len(qualified),
            orders=sorted({n for item in items for n in item.orders}),
            explained_power=share, score=score,
            rpm_min=round(60*min(values), 2), rpm_max=round(60*max(values), 2),
            bands=[list(band) for band in sorted({b for item in items for b in item.bands})],
            blade_pass_hypotheses=bpf,
            track=[dict(window=i, time_seconds=round(centers[i], 3), hz=round(e.hz, 3),
                        rpm=round(e.hz*60, 2), orders=list(e.orders)) for i, _, e in track]))
    ranked.sort(key=lambda x: (-x["score"], -x["qualified_coverage"], x["hz"]))
    accepted = []
    if ranked:
        best = ranked[0]
        runner = ranked[1]["score"] if len(ranked) > 1 else 0
        if (best["qualified_coverage"] >= cfg["min_window_fraction"]
                and best["score"] >= cfg["min_score"]
                and best["score"]-runner >= cfg["min_score_margin"]):
            accepted = [dict(best, rpm=round(60*best["hz"], 2),
                             assumption="one acoustic fundamental period equals one shaft revolution")]
    return ranked[:limit], accepted


def blade_pass_rpm(windows, blades, cfg, centers=None, accepted=None):
    """Prefer a coherent shaft family with the specified dominant blade order.

    A lone persistent modulation can only be converted under an explicit BPF
    assumption. A conflicting coherent family is not silently divided.
    """
    if accepted is None:
        _, accepted = rank_windows(windows, cfg, centers)
    if accepted and cfg.get("tracking", True):
        best = accepted[0]
        hypothesis = next((h for h in best.get("blade_pass_hypotheses", [])
                           if h["blade_count"] == blades), None)
        if hypothesis is None:
            return []
        return [dict(best, blade_count=blades, blade_pass_hz=hypothesis["hz"],
                     blade_basis="tracked_harmonic", assumption="specified blade count and observed dominant harmonic interpreted as BPF")]
    # Compatibility: single strong modulation, with an explicit user assumption.

    groups = dominant_groups(windows, cfg)
    if not groups or (len(groups) > 1 and len(groups[0]) == len(groups[1])):
        return []
    group = groups[0]
    coverage = len(group) / len(windows)
    if coverage < cfg["min_window_fraction"]:
        return []
    blade_hz = float(np.median([p[0] for p in group]))
    hz = blade_hz / blades
    return [dict(hz=round(hz, 3), rpm=round(hz * 60, 2), segments=len(group),
                 coverage=coverage, blade_count=blades, blade_pass_hz=round(blade_hz, 3),
                 rpm_min=round(60*min(p[0] for p in group)/blades, 2),
                 rpm_max=round(60*max(p[0] for p in group)/blades, 2),
                 blade_basis="assumed_dominant", assumption="dominant acoustic modulation assumed to be BPF")]

def dominant_groups(windows, cfg):
    """Group detected dominant modulation peaks by their frequency resolution."""
    peaks = sorted((hz, res) for hz, res, _ in windows if hz is not None)
    groups = []
    for hz, res in peaks:
        if not groups or hz - groups[-1][0][0] > cfg["match_bins"] * min(res, groups[-1][0][1]):
            groups.append([])
        groups[-1].append((hz, res))
    return sorted(groups, key=len, reverse=True)


def longest_rpm_run(track, ranges, cfg):
    """Count adjacent windows with a fundamental and co-occurring harmonics."""
    longest = []
    current = []
    for point in track:
        if 1 not in point["orders"] or len(point["orders"]) < cfg["min_rpm_orders"]:
            current = []
            continue
        index = point["window"]
        if current and index != current[-1] + 1:
            current = []
        current.append(index)
        if len(current) > len(longest):
            longest = current.copy()
    if not longest:
        return dict(windows=0, start=None, end=None, seconds=0.0)
    start = ranges[longest[0]][0]
    end = ranges[longest[-1]][1]
    return dict(windows=len(longest), start=start, end=end, seconds=end - start)


def clip_shaft_evidence(windows, ranges, cfg):
    """Rank clip evidence by observed windows and present conditional conversions."""
    centers = [(start + end) / 2 for start, end in ranges]
    candidates, _ = rank_windows(windows, cfg, centers, limit=12)
    for candidate in candidates:
        candidate["continuity"] = longest_rpm_run(candidate["track"], ranges, cfg)

    def support(candidate):
        return (candidate["qualified_segments"], candidate["continuity"]["seconds"],
                candidate["segments"])

    candidates.sort(key=lambda candidate: tuple(-value for value in support(candidate)) + (candidate["hz"],))
    threshold = cfg["min_contiguous_windows"]
    minimum_seconds = threshold * cfg["window_seconds"]
    eligible = [c for c in candidates if c["continuity"]["windows"] >= threshold
                and c["continuity"]["seconds"] >= minimum_seconds - 1e-6]

    estimate = []
    if eligible and (len(eligible) == 1 or support(eligible[0]) != support(eligible[1])):
        best = eligible[0]
        run = best["continuity"]
        run_hz = [point["hz"] for point in best["track"]
                  if ranges[point["window"]][0] >= run["start"]
                  and ranges[point["window"]][1] <= run["end"]]
        hz = float(np.median(run_hz))
        estimate = [dict(best, hz=round(hz, 3), rpm=round(hz * 60, 2),
                         assumption="one acoustic fundamental period equals one shaft revolution")]

    reference = None
    if candidates:
        best = candidates[0]
        reference = dict(hz=best["hz"], rpm=round(best["hz"] * 60, 2),
                         segments=best["segments"], kind="harmonic")
    else:
        groups = dominant_groups(windows, cfg)
        if groups:
            group = groups[0]
            hz = float(np.median([peak for peak, _ in group]))
            reference = dict(hz=round(hz, 3), rpm=round(hz * 60, 2),
                             segments=len(group), kind="dominant")
    if estimate:
        top = eligible[0]
        candidates = [top, *(item for item in candidates if item is not top)]
    return candidates[:3], estimate, reference

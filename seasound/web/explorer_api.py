"""Recording catalog, safe WAV playback, waveform overview and signal measurements.

The existing /api/stft blueprint is intentionally kept separate.
"""
import csv
import hashlib
import math
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
from flask import Blueprint, current_app, jsonify, request, send_file
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, periodogram, sosfiltfilt, welch

explorer_bp = Blueprint('recordings', __name__, url_prefix='/api/recordings')
_catalog_cache = {}
_catalog_lock = threading.Lock()


def audio_root():
    return Path(current_app.config['AUDIO_ROOT']).expanduser().resolve()


def resolve_recording(value):
    value = str(value).replace('\\', '/')
    relative = Path(value)
    if not value or relative.is_absolute() or ':' in value or '..' in relative.parts:
        raise ValueError('녹음 폴더 내부의 상대 WAV 경로를 지정하세요.')
    root = audio_root()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target.suffix.lower() != '.wav':
        raise ValueError('허용된 WAV 경로가 아닙니다.')
    if not target.is_file():
        raise FileNotFoundError('WAV 파일을 찾지 못했습니다.')
    return target


def number(key, default, low, high):
    try:
        value = float(request.args.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f'{key}: 올바른 숫자를 입력하세요.')
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{key}: {low}~{high} 범위로 입력하세요.')
    return value


@explorer_bp.errorhandler(ValueError)
def bad_request(error):
    return jsonify(error=str(error)), 400


@explorer_bp.errorhandler(FileNotFoundError)
def not_found(error):
    return jsonify(error=str(error)), 404


@explorer_bp.errorhandler(sf.LibsndfileError)
def bad_audio(error):
    return jsonify(error='WAV 파일을 해석하지 못했습니다. 파일 형식을 확인하세요.'), 422


def catalog(refresh=False):
    root = audio_root()
    manifest_name = current_app.config.get('RECORDINGS_CSV', '')
    stamp = Path(manifest_name).stat().st_mtime_ns if manifest_name and Path(manifest_name).is_file() else 0
    key = (str(root), str(manifest_name), stamp)
    with _catalog_lock:
        cached = _catalog_cache.get(key)
        if not refresh and cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        overrides, notices = {}, []
        if manifest_name:
            if not Path(manifest_name).is_file():
                notices.append('메타데이터 CSV를 찾지 못해 폴더 구조로 분류했습니다.')
            else:
                with open(manifest_name, encoding='utf-8-sig', newline='') as handle:
                    reader = csv.DictReader(handle)
                    fields = set(reader.fieldnames or [])
                    label = 'class' if 'class' in fields else 'unified_label'
                    if not {'path', 'dataset', label}.issubset(fields):
                        raise ValueError('CSV에 path, dataset, class 또는 unified_label이 필요합니다.')
                    for row in reader:
                        relative = Path(row['path'].replace(chr(92), '/')).as_posix()
                        overrides[relative] = {'dataset': row['dataset'], 'class': row[label]}
        items, skipped = [], 0
        if root.is_dir():
            for candidate in sorted(root.rglob('*')):
                if candidate.suffix.lower() != '.wav' or not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(root):
                        continue
                    rel = candidate.relative_to(root).as_posix()
                    info = sf.info(resolved)
                    if info.frames <= 0 or info.samplerate <= 0:
                        skipped += 1
                        continue
                    parts = Path(rel).parts
                    meta = overrides.get(rel, {})
                    dataset = meta.get('dataset') or (parts[0] if len(parts) >= 3 else '미지정')
                    category = meta.get('class') or (parts[-2] if len(parts) >= 2 else '미지정')
                    items.append(dict(path=rel, name=candidate.name, dataset=dataset,
                                      **{'class': category}, duration=round(info.duration, 6),
                                      sample_rate=info.samplerate, channels=info.channels))
                except (RuntimeError, OSError, ValueError):
                    skipped += 1
        else:
            notices.append('녹음 폴더가 없습니다. SEASOUND_AUDIO_ROOT를 실제 WAV 폴더로 설정하세요.')
        if skipped:
            notices.append(f'읽을 수 없는 녹음 {skipped}개를 목록에서 제외했습니다.')
        result = dict(items=items, notices=notices)
        # Bound this process-local cache when configurations change during tests.
        if len(_catalog_cache) > 8:
            _catalog_cache.clear()
        _catalog_cache[key] = (time.monotonic(), result)
        return result


@explorer_bp.get('', strict_slashes=False)
def recordings():
    result = catalog(request.args.get('refresh') == '1')
    dataset, category = request.args.get('dataset', ''), request.args.get('class', '')
    query = request.args.get('q', '').strip().casefold()
    lo, hi = number('min_duration', 0, 0, 1e9), number('max_duration', 1e9, 0, 1e9)
    if lo > hi:
        raise ValueError('최소 길이는 최대 길이보다 작거나 같아야 합니다.')
    offset, limit = int(number('offset', 0, 0, 1e9)), int(number('limit', 100, 1, 200))
    all_items = result['items']
    items = [r for r in all_items if (not dataset or r['dataset'] == dataset)
             and (not category or r['class'] == category) and lo <= r['duration'] <= hi
             and (not query or query in (r['path'] + ' ' + r['dataset'] + ' ' + r['class']).casefold())]
    return jsonify(recordings=items[offset:offset + limit], total=len(items),
                   offset=offset, limit=limit, notices=result['notices'],
                   datasets=sorted({r['dataset'] for r in all_items}),
                   classes=sorted({r['class'] for r in all_items if not dataset or r['dataset'] == dataset}))


@explorer_bp.get('/audio')
def audio():
    path = resolve_recording(request.args.get('path', ''))
    info = sf.info(path)
    # Browsers consistently decode PCM16 WAV. Preserve sample rate and all channels.
    if info.subtype == 'PCM_16':
        return send_file(path, mimetype='audio/wav', conditional=True)
    stat = path.stat()
    key = hashlib.sha256(f'{path}:{stat.st_mtime_ns}:{stat.st_size}'.encode()).hexdigest()
    folder = Path(current_app.config['AUDIO_PLAYBACK_CACHE'])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{key}.wav'
    if not target.is_file():
        fd, temporary = tempfile.mkstemp(suffix='.wav', dir=folder)
        os.close(fd)
        try:
            with sf.SoundFile(path) as source, sf.SoundFile(temporary, 'w', samplerate=info.samplerate,
                    channels=info.channels, subtype='PCM_16', format='WAV') as destination:
                for block in source.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    destination.write(np.clip(block, -1, 1))
            os.replace(temporary, target)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    return send_file(target, mimetype='audio/wav', conditional=True, download_name=path.stem + '.wav')


@explorer_bp.get('/waveform')
def waveform():
    path = resolve_recording(request.args.get('path', ''))
    with sf.SoundFile(path) as source:
        count = min(source.frames, 1200)
        if not count:
            raise ValueError('빈 녹음입니다.')
        edges = np.linspace(0, source.frames, count + 1, dtype=np.int64)
        minima, maxima, times = [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            # Bounded memory even for long, high-rate recordings.
            remaining, low, high = int(right - left), float('inf'), float('-inf')
            while remaining:
                block = source.read(min(remaining, 65536), dtype='float32', always_2d=True)
                if not len(block):
                    raise ValueError('WAV 파일 길이가 올바르지 않습니다.')
                if not np.isfinite(block).all():
                    raise ValueError('WAV에 유효하지 않은 신호 값이 있습니다.')
                mono = block.mean(axis=1)
                low, high = min(low, float(mono.min())), max(high, float(mono.max()))
                remaining -= len(block)
            minima.append(low)
            maxima.append(high)
            times.append(float((left + right) / 2 / source.samplerate))
        return jsonify(duration=source.frames / source.samplerate, sample_rate=source.samplerate,
                       times=times, min=minima, max=maxima, channel_mix='mean')


def harmonic_families(envelope, rate):
    """Find observed fundamentals with visible second and third harmonics."""
    freq, power = periodogram(envelope, fs=rate, window='hann', detrend='constant')
    step = freq[1]
    band = (freq >= .5) & (freq <= 50)
    if not np.any(band):
        return []
    reference = power[band].max()
    if reference <= 0:
        return []
    db = 10 * np.log10(np.maximum(power / reference, 1e-12))
    peaks, _ = find_peaks(db, prominence=9)
    floor = np.median(db[band])
    peaks = [p for p in peaks if band[p] and db[p] >= max(-26, floor + 10)]
    # Sub-bin peak locations using the log-power parabola at a Hann peak.
    lines = []
    for p in peaks:
        curvature = db[p-1] - 2 * db[p] + db[p+1]
        shift = np.clip((db[p-1] - db[p+1]) / (2 * curvature), -.5, .5) if curvature else 0
        lines.append(float(freq[p] + shift * step))
    candidates = []
    for base in lines:
        if base < max(.5, 3 * step) or 3 * base > 50:
            continue
        matches = [base]
        for order in (2, 3):
            target = order * base
            nearest = min(lines, key=lambda v: abs(v-target))
            if abs(nearest-target) <= max(.65 * step, .015 * target):
                matches.append(nearest)
        # A second comparably strong, unrelated line makes the source ambiguous.
        def belongs_to_family(line):
            order = round(line / base)
            return order >= 1 and abs(line - order * base) <= max(step, .02 * line)
        competing = any(db[p] >= -8 and not belongs_to_family(line)
                        for p, line in zip(peaks, lines))
        if len(matches) == 3 and not competing:
            candidates.append(float(np.median(np.asarray(matches) / [1, 2, 3])))
    return candidates


def recurring_fundamental(envelope, rate):
    """Require the same three-line family in two non-overlapping intervals."""
    first, second = np.array_split(envelope, 2)
    left = harmonic_families(first, rate)
    right = harmonic_families(second, rate)
    resolution = rate / len(first)
    matches = [(a + b) / 2 for a in left for b in right
               if abs(a-b) <= max(.55 * resolution, .025 * a)]
    return matches[0] if len(matches) == 1 else None


def measure_signal(samples, rate, band_low, band_high, blades=None):
    """Welch tonals and an acoustic shaft candidate from periodic modulation."""
    x = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('WAV에 유효하지 않은 신호 값이 있습니다.')
    x = x - x.mean()
    nperseg = min(len(x), int(rate * 4), 65536)
    f, psd = welch(x, rate, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    reference = float(psd.max())
    empty = dict(tonals=[], modulation_hz=None, shaft_hz=None, shaft_rpm=None,
                 blade_count=blades, resolution_hz=float(f[1] - f[0]), band_hz=[band_low, band_high],
                 note='변조 후보를 검출하지 못했습니다. 회전수를 추정하지 않습니다.')
    if reference <= 1e-20 or np.sqrt(np.mean(x*x)) < 1e-9:
        empty['note'] = '무음 또는 신호가 너무 약합니다.'
        return empty
    db = 10 * np.log10(np.maximum(psd / reference, 1e-12))
    peaks, properties = find_peaks(db, prominence=8)
    floor = median_filter(db, size=31, mode='nearest')
    candidates = [int(p) for p in peaks if f[p] >= 10 and db[p] - floor[p] >= 6 and db[p] >= -50]
    candidates.sort(key=lambda p: psd[p], reverse=True)
    empty['tonals'] = [dict(hz=round(float(f[p]), 3), relative_db=round(float(db[p]), 2)) for p in candidates[:5]]
    filtered = sosfiltfilt(butter(4, [band_low, band_high], btype='bandpass', fs=rate, output='sos'), x)
    # Trim filter/Hilbert edges; reject near-constant envelopes such as a pure tone.
    envelope = np.abs(hilbert(filtered))
    trim = min(int(rate * .25), len(envelope) // 8)
    envelope = envelope[trim:-trim] if trim else envelope
    mean = float(envelope.mean())
    if mean <= 1e-12 or float(envelope.std()) / mean < .02:
        return empty
    fm, pm = welch(envelope - mean, rate, window='hann',
                   nperseg=min(len(envelope), int(rate * 4), 65536))
    valid = np.flatnonzero((fm >= .5) & (fm <= 50))
    if len(valid) < 3:
        return empty
    pm_db = 10 * np.log10(np.maximum(pm / max(float(pm.max()), 1e-30), 1e-12))
    peaks, props = find_peaks(pm_db, prominence=8)
    selected = [int(p) for p in peaks if .5 <= fm[p] <= 50 and pm[p] > 10 * np.median(pm[valid])]
    if not selected:
        return empty
    peak = max(selected, key=lambda p: pm[p])
    hz = float(fm[peak])
    empty.update(modulation_hz=round(hz, 3),
                 note='포락선의 우세 변조 후보입니다. 블레이드 통과 주파수라는 검증이 필요합니다.')
    shaft = None if blades else recurring_fundamental(envelope, rate)
    if shaft is not None:
        empty.update(shaft_hz=round(shaft, 3), shaft_rpm=round(shaft * 60, 2),
                     shaft_method='envelope_harmonics',
                     note='앞뒤 구간의 기본 주파수와 2·3배음이 반복된 음향 후보입니다.')
    elif blades:
        # Preserve the explicit, caller-supplied blade-pass mode for older API users.
        empty.update(shaft_hz=round(hz / blades, 3), shaft_rpm=round(hz / blades * 60, 2),
                     shaft_method='explicit_blade_pass',
                     note=f'변조 후보를 블레이드 통과 주파수로 가정하고 {blades}개 블레이드로 나눈 조건부 추정입니다.')
    else:
        empty['note'] = '두 구간에서 일관된 배음 간격을 식별하지 못했습니다.'
    return empty


@explorer_bp.get('/analysis')
def analysis():
    path = resolve_recording(request.args.get('path', ''))
    start, seconds = number('start', 0, 0, 1e9), number('seconds', 10, 2, 30)
    blade_value = request.args.get('blades', '').strip()
    blades = number('blades', 1, 1, 20) if blade_value else None
    if blades is not None and not blades.is_integer():
        raise ValueError('블레이드 수는 정수여야 합니다.')
    blades = int(blades) if blades else None
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        low = number('band_low', 200, 1, 1000000)
        high = number('band_high', min(3000, rate * .45), 1, 1000000)
        if not low < high < rate / 2:
            raise ValueError(f'복조 대역은 0 < 하한 < 상한 < {rate / 2:g} Hz여야 합니다.')
        frame = int(start * rate)
        if frame >= source.frames:
            raise ValueError('분석 시작 시간이 녹음 길이를 벗어났습니다.')
        frames = min(int(seconds * rate), source.frames - frame)
        if frames < 2 * rate:
            raise ValueError('변조 분석에는 최소 2초의 구간이 필요합니다.')
        if frames * source.channels > 8000000:
            raise ValueError('분석 데이터가 너무 큽니다. 구간 길이를 줄이세요.')
        source.seek(frame)
        x = source.read(frames, dtype='float64', always_2d=True).mean(axis=1)
    result = measure_signal(x, rate, low, high, blades)
    if path.parent.name.lower() == 'background':
        result.update(shaft_hz=None, shaft_rpm=None, shaft_method=None, note='배경 녹음')
    result.update(start=frame / rate, end=(frame + frames) / rate, method='Welch PSD / bandpass + Hilbert envelope + Welch',
                  channel_mix='mean')
    return jsonify(result)

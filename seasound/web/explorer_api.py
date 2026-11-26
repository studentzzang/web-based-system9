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
from scipy.signal import find_peaks, welch

from dsp import demon
from dsp.explorer_spectrogram import MODES, recording_spectrogram
from dsp.tonal import TONAL_BAND_HZ, representative_tonal

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


def measure_signal(samples, rate, bounds, band_limits, cfg, blades=None):
    """Welch tonal peaks plus conditional RPM evidence for the segment API."""
    x = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('WAV에 유효하지 않은 신호 값이 있습니다.')
    x = x - x.mean()
    nperseg = min(len(x), int(rate * 4), 65536)
    f, psd = welch(x, rate, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    reference = float(psd.max())
    empty = dict(tonals=[], modulation_hz=None, shaft_hz=None, shaft_rpm=None,
                 blade_count=blades, resolution_hz=float(f[1] - f[0]), band_hz=list(band_limits), shaft_candidates=[],
                 note='변조 후보를 검출하지 못했습니다. 회전수를 추정하지 않습니다.')
    if reference <= 1e-20 or np.sqrt(np.mean(x*x)) < 1e-9:
        empty['note'] = '무음 또는 신호가 너무 약합니다.'
        return empty
    db = 10 * np.log10(np.maximum(psd / reference, 1e-12))
    peaks, properties = find_peaks(db, prominence=8)
    floor = median_filter(db, size=31, mode='nearest')
    candidates = [int(p) for p in peaks if f[p] >= 10 and db[p] - floor[p] >= 6 and db[p] >= -50]
    candidates.sort(key=lambda p: psd[p], reverse=True)
    empty['tonals'] = [dict(hz=float(f[p]), relative_db=round(float(db[p]), 2)) for p in candidates[:5]]
    if len(x) < 2 * rate:
        return empty
    window = demon.analyze_window(x, rate, cfg, bounds)
    dominant = window[0]
    empty["modulation_hz"] = round(dominant, 3) if dominant is not None else None
    candidates, accepted = demon.rank_windows([window], cfg)
    empty["shaft_candidates"] = candidates
    if blades and dominant is not None:
        empty.update(shaft_hz=dominant/blades, shaft_rpm=round(60*dominant/blades, 2),
                     shaft_method="explicit_blade_pass",
                     note="우세 변조가 날개 통과음이라는 가정으로 환산했습니다.")
    elif accepted:
        empty.update(shaft_hz=accepted[0]["hz"], shaft_rpm=accepted[0]["rpm"],
                     shaft_method="multiband_harmonics",
                     note="기본 반복을 축 1회전으로 가정한 음향 후보입니다.")
    elif dominant is not None:
        empty["note"] = "변조 검출 · 축 RPM 근거 부족"
    return empty


def clip_analysis(source, cfg, bounds, background, blades):
    rate = source.samplerate
    if source.frames < 2:
        raise ValueError('분석할 음향 샘플이 부족합니다.')
    windows, measurements = [], []
    while source.tell() < source.frames:
        first = source.tell()
        remaining = source.frames - first
        frames = remaining if remaining < 22 * rate else 20 * rate
        windows.append((first, first + frames))
        samples = source.read(frames, dtype="float64", always_2d=True).mean(axis=1)
        if len(samples) != frames:
            raise ValueError('WAV 파일 길이가 올바르지 않습니다.')
        if not background and frames >= 2 * rate:
            measurements.append(demon.analyze_window(samples, rate, cfg, bounds))
    candidates, shaft_groups = demon.rank_windows(measurements, cfg)
    method = "multiband_harmonics"
    if blades and not background:
        method = "explicit_blade_pass"
        shaft_groups = demon.blade_pass_rpm(measurements, blades, cfg)
    return {
        "scope": "clip", "start": 0, "end": source.frames / rate,
        "segments": len(windows), "segment_seconds": 20, "background": background,
        "shaft_candidates": candidates, "shaft_method": method,
        "modulated_segments": sum(w[0] is not None for w in measurements),
        "tonals": representative_tonal(source, windows), "tonal_band_hz": TONAL_BAND_HZ,
        "shafts": shaft_groups,
    }


def is_background_recording(path):
    relative = path.relative_to(audio_root()).as_posix()
    for record in catalog()["items"]:
        if record["path"] == relative:
            return record["class"].strip().casefold() == "background"
    return path.parent.name.casefold() == "background"


@explorer_bp.get('/analysis')
def analysis():
    path = resolve_recording(request.args.get('path', ''))
    scope = request.args.get('scope', 'segment')
    if scope not in ('segment', 'clip'):
        raise ValueError('지원하지 않는 분석 범위입니다.')
    start, seconds = number('start', 0, 0, 1e9), number('seconds', 10, 2, 30)
    blade_value = request.args.get('blades', '').strip()
    blades = number('blades', 1, 1, 20) if blade_value else None
    if blades is not None and not blades.is_integer():
        raise ValueError('블레이드 수는 정수여야 합니다.')
    blades = int(blades) if blades else None
    background = is_background_recording(path)
    cfg = demon.read_settings()
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        low = number('band_low', cfg['carrier_bands_hz'][0][0], 0, rate / 2)
        high = number('band_high', min(rate / 2, cfg['carrier_bands_hz'][-1][1]), 0, rate / 2)
        if low >= high:
            raise ValueError(f'복조 대역은 0 ≤ 하한 < 상한 ≤ {rate / 2:g} Hz여야 합니다.')
        bounds = (low, high) if 'band_low' in request.args or 'band_high' in request.args else None
        if scope == 'clip':
            return jsonify(clip_analysis(source, cfg, bounds, background, blades))
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
    result = measure_signal(x, rate, bounds, (low, high), cfg, blades)
    for tonal in result['tonals']:
        tonal['hz'] = round(tonal['hz'], 3)
    if result['shaft_hz'] is not None:
        result['shaft_hz'] = round(result['shaft_hz'], 3)
    if background:
        result.update(shaft_hz=None, shaft_rpm=None, shaft_method=None, shaft_candidates=[], note='배경 녹음')
    result.update(start=frame / rate, end=(frame + frames) / rate, method='Welch PSD / Hilbert envelope + periodogram',
                  channel_mix='mean')
    return jsonify(result)


@explorer_bp.get('/spectrogram')
def spectrogram():
    mode = request.args.get('mode', '').lower()
    if mode not in MODES:
        raise ValueError('지원하지 않는 시각화 모드입니다.')
    path = resolve_recording(request.args.get('path', ''))
    return jsonify(recording_spectrogram(path, mode))


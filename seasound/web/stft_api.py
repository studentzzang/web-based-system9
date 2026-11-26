
import numpy as np
import soundfile as sf
from flask import Blueprint, current_app, jsonify, request

from dsp.stft_cache import load_or_compute_stft


stft_bp = Blueprint("stft", __name__)


def reduce_spectrogram(
    frequencies,
    times,
    power,
    max_freq_bins=256,
    max_time_bins=600,
):
    # 원본 축을 몇 개 구간으로 나눌지 결정
    freq_count = min(len(frequencies), max_freq_bins)
    time_count = min(len(times), max_time_bins)

    freq_edges = np.linspace(
        0, len(frequencies), freq_count + 1, dtype=int
    )
    time_edges = np.linspace(
        0, len(times), time_count + 1, dtype=int
    )

    # 각 구간에서 최대 파워를 선택
    reduced = np.maximum.reduceat(
        power, freq_edges[:-1], axis=0
    )
    reduced = np.maximum.reduceat(
        reduced, time_edges[:-1], axis=1
    )

    # 축 좌표를 구간 중심으로 변경
    freq_axis = (
        frequencies[freq_edges[:-1]]
        + frequencies[freq_edges[1:] - 1]
    ) / 2

    time_axis = (
        times[time_edges[:-1]]
        + times[time_edges[1:] - 1]
    ) / 2

    # 녹음 전체의 최대 파워를 기준으로 상대 dB 변환
    reference = float(np.max(power))

    if reference > 0:
        db = 10 * np.log10(
            np.maximum(reduced / reference, 1e-12)
        )
    else:
        db = np.full(reduced.shape, -120.0)

    return freq_axis, time_axis, db


@stft_bp.get("/api/stft")
def get_stft():
    path = request.args.get("path", "").strip()

    if not path:
        return jsonify({"error": "WAV 경로가 필요합니다."}), 400

    try:
        frequencies, times, power, cache_hit = (
            load_or_compute_stft(path, audio_root=current_app.config.get("AUDIO_ROOT"))
        )

        freq_axis, time_axis, db = reduce_spectrogram(
            frequencies, times, power,
            max_freq_bins=len(frequencies) if request.args.get("detail") == "frequency" else 256,
        )

    except FileNotFoundError:
        return jsonify({"error": "WAV 파일을 찾지 못했습니다."}), 404

    except sf.LibsndfileError:
        return jsonify({"error": "WAV 파일을 해석하지 못했습니다."}), 422

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({
        "path": path,
        "cache_hit": cache_hit,
        "frequencies": np.round(freq_axis, 2).tolist(),
        "times": np.round(time_axis, 3).tolist(),
        "db": np.round(db.astype(np.float64), 2).tolist(),
        "shape": list(db.shape),
    })

# -*- coding: utf-8 -*-
"""dsp/mel.py — 세그먼트 분할과 log-mel 계산."""
import librosa
import numpy as np


def segment_starts(n_samples, sr, length_sec, hop_sec):
    """5초 창을 2.5초씩 밀 때의 시작 샘플 위치. 남는 꼬리는 버린다."""
    length = round(length_sec * sr)
    hop = round(hop_sec * sr)
    if n_samples < length:
        return []
    n_seg = (n_samples - length) // hop + 1
    return [i * hop for i in range(n_seg)]


def logmel(audio, sr, n_fft=1024, hop_length=320, n_mels=128,
           fmin=10.0, fmax=8000.0, ref=1.0, amin=1e-10, top_db=None):
    """(n_mels, frames) float32 log-mel. ref=1.0이면 절대 레벨(dB), "max"면 세그먼트 최대값 기준."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax, power=2.0, center=True, window="hann")
    ref_value = np.max if ref == "max" else float(ref)
    return librosa.power_to_db(mel, ref=ref_value, amin=amin, top_db=top_db).astype(np.float32)

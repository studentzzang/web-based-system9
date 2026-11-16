
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_rate: int = 16000,
) -> tuple[np.ndarray, int]:

    if audio.ndim != 1:
        raise ValueError("1차원 Mono 오디오만 지원합니다.")

    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError("샘플링레이트는 양수여야 합니다.")

    if len(audio) == 0:
        raise ValueError("빈 오디오는 처리할 수 없습니다.")

    # 이미 목표 샘플링레이트라면 변환하지 않음
    if sample_rate == target_rate:
        return audio.astype(np.float32, copy=False), target_rate

    # 리샘플링 비율 계산
    factor = gcd(sample_rate, target_rate)

    up = target_rate // factor
    down = sample_rate // factor

    # 안티에일리어싱 필터를 포함한 리샘플링
    resampled = resample_poly(
        audio,
        up=up,
        down=down,
    )

    return resampled.astype(np.float32, copy=False), target_rate

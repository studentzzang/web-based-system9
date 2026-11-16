
import numpy as np
from scipy.signal import stft


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if audio.ndim != 1:
        raise ValueError("Mono 오디오만 지원합니다.")

    if sample_rate != 16000:
        raise ValueError("16kHz 오디오만 지원합니다.")

    if not (0 < hop_length <= n_fft):
        raise ValueError("FFT 및 Hop 설정이 잘못되었습니다.")

    if len(audio) < n_fft:
        raise ValueError("STFT를 계산하기에 오디오가 너무 짧습니다.")

    # STFT 계산
    frequencies, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
        detrend=False,
        scaling="spectrum",
    )

    # 복소수 STFT -> 파워 스펙트럼
    power = (np.abs(spectrum) ** 2).astype(np.float32)

    return frequencies, times, power

"""Bounded spectrogram data for Explorer's supplementary views."""
import math

import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt, stft

from data_io.preprocess import resample_audio


BLOCK_SECONDS = 20


def logmel_chunk(samples, rate):
    import librosa
    from dsp.mel import logmel

    mono, rate = resample_audio(samples, rate)
    levels = logmel(mono, rate, ref=1.0, amin=1e-30)
    times = np.arange(levels.shape[1]) * .02
    centers = librosa.mel_frequencies(n_mels=130, fmin=10, fmax=8000)[1:-1]
    return centers, times, levels


def lofar_chunk(samples, rate):
    mono, rate = resample_audio(samples, rate, target_rate=2000)
    mono = mono - mono.mean()
    if len(mono) < rate:
        mono = np.pad(mono, (0, rate - len(mono)))
    f, t, z = stft(mono, fs=rate, nperseg=rate, noverlap=3 * rate // 4)
    low = f <= 500
    return f[low], t, power_db(np.abs(z[low]) ** 2)


def demon_upper_hz(rate):
    return min(8000, rate / 2)


def demon_envelope(samples, rate, low, high):
    # The DC offset is not an acoustic carrier. At Nyquist no upper filter is needed.
    signal = np.asarray(samples, dtype=np.float64)
    signal = signal - signal.mean()
    nyquist = rate / 2
    if 0 < low and high < nyquist:
        sos = butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    elif high < nyquist:
        sos = butter(4, high, btype="lowpass", fs=rate, output="sos")
    elif low > 0:
        sos = butter(4, low, btype="highpass", fs=rate, output="sos")
    else:
        sos = None
    if sos is not None:
        signal = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(signal))
    trim = min(rate // 4, len(envelope) // 8)
    return envelope[trim:-trim] if trim else envelope


def power_db(power):
    return 10 * np.log10(np.maximum(power, 1e-30))


def demon_chunk(samples, rate):
    envelope = demon_envelope(samples, rate, 0, demon_upper_hz(rate))
    envelope, env_rate = resample_audio(envelope, rate, target_rate=200)
    f, power = periodogram(envelope, fs=env_rate, window="hann",
                           nfft=BLOCK_SECONDS * env_rate, detrend="constant")
    if envelope.std() < max(1e-12, .02 * envelope.mean()):
        power[:] = 0
    band = f <= 50
    return f[band], np.zeros(1), power_db(power[band, None])


MODES = {
    "logmel": (logmel_chunk, .02, "mel_center_hz"),
    "lofar": (lofar_chunk, .25, "frequency_hz"),
    "demon": (demon_chunk, BLOCK_SECONDS, "modulation_hz"),
}


def recording_spectrogram(path, mode):
    transform, hop, axis = MODES[mode]
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        duration = source.frames / rate
        if not duration:
            raise ValueError("빈 녹음입니다.")

        if mode == "demon" and duration < 2:
            raise ValueError("DEMON 분석에는 최소 2초의 녹음이 필요합니다.")
        width = min(600, math.ceil(duration / hop))
        time_sum = np.zeros(width)
        time_count = np.zeros(width, dtype=int)
        output = None
        for first in range(0, source.frames, BLOCK_SECONDS * rate):
            last = min(source.frames, first + BLOCK_SECONDS * rate)
            begin = max(0, first - rate)
            end = min(source.frames, last + rate)
            if mode == "demon":
                # For a short tail, include up to two seconds ending at EOF.
                begin, end = min(first, last - 2 * rate), last
            source.seek(begin)
            samples = source.read(end - begin, dtype="float32", always_2d=True).mean(axis=1)
            if not np.isfinite(samples).all():
                raise ValueError("WAV에 유효하지 않은 신호 값이 있습니다.")

            frequencies, times, levels = transform(samples, rate)
            times += (first + last) / (2 * rate) if mode == "demon" else begin / rate
            inside = (times >= first / rate) & (times < last / rate)
            frame_times = times[inside]
            columns = np.minimum(np.round(frame_times * width / duration, 9).astype(int), width - 1)
            if output is None:
                output = np.full((len(frequencies), width), -np.inf)
            np.maximum.at(output.T, columns, levels[:, inside].T)
            np.add.at(time_sum, columns, frame_times)
            np.add.at(time_count, columns, 1)

    populated = time_count > 0
    if not populated.any():
        raise ValueError("스펙트로그램을 계산할 수 있는 구간이 없습니다.")
    output = output[:, populated]
    peak = float(output.max())
    db = np.clip(output - peak, -120, 0) if peak > -299.99 else np.full_like(output, -120)
    times = time_sum[populated] / time_count[populated]
    return {
        "mode": mode, "duration": duration, "axis": axis,
        "frequencies": frequencies.round(6).tolist(), "times": times.round(6).tolist(),
        "db": db.round(2).tolist(), "shape": list(db.shape),
    }

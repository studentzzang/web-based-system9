
import unittest

import numpy as np

from data_io.loader import load_wav
from data_io.preprocess import resample_audio
from dsp.stft import compute_stft


class TestSTFT(unittest.TestCase):

    def test_real_wav(self):
        # 1. WAV 로딩
        audio, sr = load_wav(
            "vessel/0.wav",
            start_sec=0,
            duration_sec=5,
        )

        # 2. 16kHz 리샘플링
        audio, sr = resample_audio(audio, sr)

        # 3. STFT 계산
        frequencies, times, power = compute_stft(audio, sr)

        print("\n주파수 빈:", len(frequencies))
        print("시간 프레임:", len(times))
        print("파워 배열:", power.shape)

        self.assertEqual(power.shape, (1025, 153))
        self.assertEqual(power.dtype, np.float32)
        self.assertAlmostEqual(times[0], 0.064)

    def test_known_frequency(self):
        # 정답이 125Hz인 가상 소리 생성
        sr = 16000
        t = np.arange(sr * 5) / sr

        audio = np.sin(2 * np.pi * 125 * t).astype(np.float32)

        frequencies, times, power = compute_stft(audio, sr)

        # 시간별 파워 평균 후 가장 강한 주파수 검출
        mean_power = power.mean(axis=1)
        peak_index = np.argmax(mean_power)
        peak_frequency = frequencies[peak_index]

        print("\n검출 주파수:", peak_frequency, "Hz")

        self.assertAlmostEqual(peak_frequency, 125.0)

    def test_invalid_sample_rate(self):
        audio = np.zeros(32000, dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_stft(audio, 32000)


if __name__ == "__main__":
    unittest.main()

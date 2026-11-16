
import unittest

import numpy as np

from data_io.loader import load_wav
from data_io.preprocess import resample_audio


class TestPreprocess(unittest.TestCase):

    def test_resample(self):
        audio, sr = load_wav(
            "vessel/0.wav",
            start_sec=0,
            duration_sec=5,
        )

        result, new_sr = resample_audio(audio, sr)

        print("\n원본:", sr, "Hz,", len(audio), "samples")
        print("변환:", new_sr, "Hz,", len(result), "samples")

        self.assertEqual(new_sr, 16000)
        self.assertEqual(len(result), 80000)
        self.assertEqual(result.dtype, np.float32)

    def test_same_sample_rate(self):
        audio = np.zeros(16000, dtype=np.float32)

        result, sr = resample_audio(audio, 16000)

        self.assertEqual(sr, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_invalid_sample_rate(self):
        audio = np.zeros(100, dtype=np.float32)

        with self.assertRaises(ValueError):
            resample_audio(audio, 0)


if __name__ == "__main__":
    unittest.main()

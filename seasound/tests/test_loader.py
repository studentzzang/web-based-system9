
import unittest
from pathlib import Path

import numpy as np

from data_io.loader import load_wav


ROOT = Path(__file__).resolve().parents[1]

VESSEL_ROOT = ROOT / "data/vtuad_wav/vessel"
BACKGROUND_ROOT = ROOT / "data/vtuad_wav/background"


class TestWavLoader(unittest.TestCase):

    def check_audio(self, folder):
        if folder == "vessel":
            wav_root = VESSEL_ROOT
        else:
            wav_root = BACKGROUND_ROOT

        # 해당 폴더에서 첫 번째 WAV 파일 선택
        files = sorted(wav_root.glob("*.wav"))
        self.assertTrue(files, f"WAV 파일 없음: {wav_root}")

        path = f"{folder}/{files[0].name}"

        audio, sample_rate = load_wav(
            path=path,
            vessel_root=VESSEL_ROOT,
            background_root=BACKGROUND_ROOT,
            start_sec=0,
            duration_sec=5,
        )

        print(f"\n파일: {path}")
        print(f"샘플링레이트: {sample_rate}")
        print(f"샘플 수: {len(audio)}")
        print(f"자료형: {audio.dtype}")

        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 5 * sample_rate)

    def test_vessel(self):
        self.check_audio("vessel")

    def test_background(self):
        self.check_audio("background")

    def test_invalid_range(self):
        files = sorted(VESSEL_ROOT.glob("*.wav"))
        self.assertTrue(files)

        with self.assertRaises(ValueError):
            load_wav(
                path=f"vessel/{files[0].name}",
                vessel_root=VESSEL_ROOT,
                background_root=BACKGROUND_ROOT,
                start_sec=1_000_000,
                duration_sec=5,
            )


if __name__ == "__main__":
    unittest.main()

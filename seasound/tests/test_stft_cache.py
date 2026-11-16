
import unittest

import numpy as np

from dsp.stft_cache import load_or_compute_stft


class TestSTFTCache(unittest.TestCase):

    def test_cache_reuse(self):
        # 첫 번째 호출: 캐시가 없으면 생성
        f1, t1, p1, hit1 = load_or_compute_stft(
            "vessel/0.wav"
        )

        # 두 번째 호출: 기존 캐시 사용
        f2, t2, p2, hit2 = load_or_compute_stft(
            "vessel/0.wav"
        )

        print("\n첫 호출 캐시 사용:", hit1)
        print("두 번째 캐시 사용:", hit2)
        print("STFT 배열 크기:", p2.shape)

        self.assertTrue(hit2)

        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()

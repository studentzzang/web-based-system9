
import unittest

from flask import Flask
from web.stft_api import stft_bp


class TestSTFTAPI(unittest.TestCase):

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(stft_bp)
        app.testing = True
        self.client = app.test_client()

    def test_stft_response(self):
        response = self.client.get(
            "/api/stft",
            query_string={"path": "vessel/0.wav"},
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        print("\nHTTP 상태:", response.status_code)
        print("캐시 사용:", data["cache_hit"])
        print("전달 배열 크기:", data["shape"])

        self.assertEqual(
            len(data["frequencies"]), data["shape"][0]
        )
        self.assertEqual(
            len(data["times"]), data["shape"][1]
        )
        self.assertEqual(
            len(data["db"]), data["shape"][0]
        )
        self.assertLessEqual(data["shape"][0], 256)
        self.assertLessEqual(data["shape"][1], 600)

    def test_missing_path(self):
        response = self.client.get("/api/stft")

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

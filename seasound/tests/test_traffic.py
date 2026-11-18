"""Traffic API behavior with a small, known AIS timeline."""

import csv
import tempfile
import unittest
from pathlib import Path

from flask import Flask

from web.traffic_api import traffic_bp


FIELDS = [
    "mmsi", "timestamp_utc", "lat", "lon", "sog_kn", "cog_deg",
    "vessel_type_code", "unified_label", "vessel_name",
]


class TrafficApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "ais_clean.csv"
        rows = [
            ["123456789", "2026-06-30T00:00:00Z", "37.8", "-122.4", "8", "90", "70", "Cargo", "A"],
            ["987654321", "2026-06-30T00:04:00Z", "37.9", "-122.3", "6", "180", "80", "Tanker", "B"],
            ["123456789", "2026-06-30T00:05:00Z", "37.81", "-122.39", "8", "91", "70", "Cargo", "A"],
            ["123456789", "2026-06-30T00:40:00Z", "37.9", "-122.2", "9", "92", "70", "Cargo", "A"],
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        app = Flask(__name__)
        app.config["AIS_CSV"] = str(path)
        app.register_blueprint(traffic_bp)
        self.client = app.test_client()

    def test_meta_and_filtered_positions(self):
        meta = self.client.get("/api/traffic/meta")
        self.assertEqual(meta.status_code, 200)
        self.assertEqual(meta.json["vessel_count"], 2)
        self.assertEqual(meta.json["point_count"], 4)
        scene = self.client.get("/api/traffic/positions?at=2026-06-30T00:10:00Z&class=Cargo")
        self.assertEqual(scene.status_code, 200)
        self.assertEqual(scene.json["count"], 1)
        self.assertEqual(scene.json["vessels"][0]["mmsi"], "123456789")
        self.assertEqual(len(scene.json["vessels"][0]["trail"]), 2)

    def test_stale_positions_and_track(self):
        scene = self.client.get("/api/traffic/positions?at=2026-06-30T00:21:00Z")
        self.assertEqual(scene.status_code, 200)
        self.assertEqual(scene.json["count"], 0)
        track = self.client.get("/api/traffic/track/123456789")
        self.assertEqual(track.status_code, 200)
        self.assertEqual(len(track.json["points"]), 3)

    def test_invalid_requests(self):
        self.assertEqual(self.client.get("/api/traffic/positions?at=not-a-date").status_code, 400)
        self.assertEqual(self.client.get("/api/traffic/positions?at=2026-06-30T00:10:00Z&class=Wrong").status_code, 400)
        self.assertEqual(self.client.get("/api/traffic/track/not-a-mmsi").status_code, 400)
        self.assertEqual(self.client.get("/api/traffic/track/111111111").status_code, 404)


if __name__ == "__main__":
    unittest.main()

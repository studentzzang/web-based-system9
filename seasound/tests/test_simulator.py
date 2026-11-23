"""Numerical and API checks for the historical hydrophone simulator."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from flask import Flask

from sim.audio_mix import mix_preview, safe_sample_path
from sim.propagation import combine_levels_db, source_level_db, thorp_db_per_km, transmission_loss_db
from web.simulator_api import simulator_bp


FIELDS = ["mmsi", "timestamp_utc", "lat", "lon", "sog_kn", "cog_deg",
          "vessel_type_code", "unified_label", "vessel_name"]


class PropagationTests(unittest.TestCase):
    def test_transmission_loss_hand_example(self):
        alpha = thorp_db_per_km(1000)
        self.assertAlmostEqual(alpha, 0.0690, delta=0.001)
        self.assertAlmostEqual(transmission_loss_db(1000, 1000), 60 + alpha, places=8)
        self.assertAlmostEqual(transmission_loss_db(1000, 1000), 60.0690, delta=0.1)
        self.assertAlmostEqual(transmission_loss_db(0, 1000, 100),
                               40 + alpha * 0.1, places=8)
        self.assertAlmostEqual(combine_levels_db([100, 100]), 103.0103, places=3)
        settings = {"source_level_db": {"Cargo": 160}, "reference_speed_kn": 10,
                    "speed_db_per_doubling": 3}
        self.assertAlmostEqual(source_level_db("Cargo", 20, settings), 163)


class SimulatorApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        ais_path = root / "ais_clean.csv"
        with ais_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows([
                ["123456789", "2026-06-30T00:00:00Z", "37.8", "-122.4", "10", "90", "70", "Cargo", "A"],
                ["123456789", "2026-06-30T00:05:00Z", "37.8", "-122.399", "10", "90", "70", "Cargo", "A"],
                ["987654321", "2026-06-30T00:05:00Z", "37.9", "-122.3", "10", "90", "80", "Tanker", "B"],
            ])
        audio_root = root / "audio"
        (audio_root / "vessel").mkdir(parents=True)
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(32000) / 16000)
        sf.write(audio_root / "vessel" / "cargo.wav", tone, 16000)
        self.settings = {
            "sample_dataset": "fixture", "frequency_hz": 1000, "seed": 42, "minimum_distance_m": 100,
            "reference_speed_kn": 10, "speed_db_per_doubling": 3,
            "source_level_db": {"Cargo": 160, "Tanker": 164, "Other": 145},
            "sample_paths": {"Cargo": "vessel/cargo.wav"},
            "audio": {"seconds": 1, "sample_rate_hz": 16000,
                      "reference_rl_db": 100, "rms_at_reference": 0.08},
        }
        config_path = root / "simulator.yaml"
        config_path.write_text(yaml.safe_dump(self.settings), encoding="utf-8")
        app = Flask(__name__)
        app.config.update(AIS_CSV=str(ais_path), AUDIO_ROOT=str(audio_root),
                          SIMULATOR_CONFIG=str(config_path), SIMULATOR_RUNS_ROOT=str(root / "runs"))
        app.register_blueprint(simulator_bp)
        self.client = app.test_client()
        self.root = root
        self.audio_root = audio_root

    def test_scene_and_timeline(self):
        meta = self.client.get("/api/simulator/meta")
        self.assertEqual(meta.status_code, 200)
        self.assertEqual(meta.json["vessel_count"], 2)
        self.assertEqual(meta.json["sample_dataset"], "fixture")
        response = self.client.get("/api/simulator/scene?at=2026-06-30T00:05:00Z&lat=37.8&lon=-122.4&radius_km=5&window_minutes=30")
        self.assertEqual(response.status_code, 200)
        body = response.json
        self.assertEqual(body["vessel_count"], 1)
        self.assertEqual(body["vessels"][0]["unified_label"], "Cargo")
        self.assertGreater(body["vessels"][0]["distance_m"], 0)
        self.assertEqual(len(body["positions"]), 2)
        self.assertGreaterEqual(len(body["series"]), 2)

    def test_audio_preview_and_run_record(self):
        response = self.client.get("/api/simulator/audio?at=2026-06-30T00:05:00Z&lat=37.8&lon=-122.4&radius_km=5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        audio, rate = sf.read(io.BytesIO(response.data))
        self.assertEqual(rate, 16000)
        self.assertEqual(len(audio), 16000)
        self.assertGreater(float(np.sqrt(np.mean(audio ** 2))), 0.01)
        run_id = response.headers["X-Simulation-Run-Id"]
        log_path = self.root / "runs" / run_id / "simulation.jsonl"
        self.assertTrue(log_path.is_file())
        record = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(record["sample_sources"][0]["sample_path"], "vessel/cargo.wav")
        self.assertEqual(record["vessel_count"], 1)
        self.assertEqual(record["sample_dataset"], "fixture")
        self.assertGreaterEqual(len(record["series"]), 2)
        self.assertIn("vessels", record["series"][0])
        self.assertEqual((self.root / "runs" / run_id / "preview.wav").read_bytes(), response.data)

    def test_invalid_inputs_and_relative_sample_paths(self):
        self.assertEqual(self.client.get("/api/simulator/scene?lat=91").status_code, 400)
        self.assertEqual(self.client.get("/api/simulator/scene?radius_km=0").status_code, 400)
        self.assertEqual(self.client.get("/api/simulator/audio?at=2026-07-01T00:00:00Z").status_code, 400)
        with self.assertRaises(ValueError):
            safe_sample_path(self.audio_root, "../outside.wav")

    def test_louder_receiver_level_makes_louder_audio(self):
        vessel = {"mmsi": "123456789", "unified_label": "Cargo", "rl_db": 90}
        quiet, _ = mix_preview([vessel], self.settings["sample_paths"], self.audio_root, self.settings)
        repeated, _ = mix_preview([vessel], self.settings["sample_paths"], self.audio_root, self.settings)
        self.assertEqual(quiet, repeated)
        _combined, used = mix_preview([vessel, {**vessel, "mmsi": "222222222"}],
                                      self.settings["sample_paths"], self.audio_root, self.settings)
        self.assertEqual(len(used), 2)
        vessel["rl_db"] = 100
        loud, _ = mix_preview([vessel], self.settings["sample_paths"], self.audio_root, self.settings)
        quiet_samples, _ = sf.read(io.BytesIO(quiet))
        loud_samples, _ = sf.read(io.BytesIO(loud))
        self.assertGreater(np.sqrt(np.mean(loud_samples ** 2)),
                           2 * np.sqrt(np.mean(quiet_samples ** 2)))


if __name__ == "__main__":
    unittest.main()

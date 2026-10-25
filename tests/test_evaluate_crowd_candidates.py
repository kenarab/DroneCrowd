"""Aggregation checks: no double counting, spatial offsets and model units."""
from pathlib import Path
import json
import os
import sys
import time
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_crowd_candidates import (clip_windows, streaming_clip, streaming_mpcount,
                                     validated_inputs, write_report_atomic)
from sample_video import sha256


class CandidateAggregationTests(unittest.TestCase):
    def test_clip_edges_are_covered_and_overlaps_averaged(self):
        rgb = np.ones((448, 464, 3), np.uint8)
        output, count = streaming_clip(rgb, lambda x: np.ones((28, 28), np.float32), time.monotonic() + 5)
        self.assertEqual(count, 6)
        np.testing.assert_array_equal(output, np.ones((56, 58)))
        self.assertEqual(output.sum(), 56 * 58)

    def test_clip_spatial_offsets_with_shifted_last_window(self):
        y, x = np.indices((448, 464))
        rgb = np.stack([x % 251, y % 251, x % 251], axis=-1).astype(np.uint8)
        predict = lambda crop: crop[::8, ::8, 0].astype(np.float32)
        output, _ = streaming_clip(rgb, predict, time.monotonic() + 5)
        np.testing.assert_array_equal(output, predict(rgb))

    def test_mpcount_partial_patch_and_thousand_factor(self):
        rgb = np.zeros((528, 768, 3), np.uint8)
        output, count = streaming_mpcount(rgb, lambda x: np.full(x.shape[:2], 1000, np.float32), time.monotonic() + 5)
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(output, np.ones((528, 768)))
        self.assertEqual(output.sum(), 528 * 768)

    def test_reject_bad_geometry_nonfinite_and_expired_deadline(self):
        with self.assertRaises(ValueError):
            list(clip_windows(200, 224))
        with self.assertRaises(ValueError):
            streaming_mpcount(np.zeros((33, 32, 3), np.uint8), lambda x: x, time.monotonic() + 5)
        with self.assertRaises(ValueError):
            streaming_clip(np.zeros((224, 224, 3), np.uint8), lambda x: np.full((28, 28), np.nan), time.monotonic() + 5)
        with self.assertRaises(TimeoutError):
            streaming_clip(np.zeros((224, 224, 3), np.uint8), lambda x: x, 0)


class BatchContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="dronecrowd-batch-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        staging = self.root / "staging"
        staging.mkdir()
        self.environment = patch.dict(os.environ, {"TMPDIR": str(staging)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.frame = {"index": 0, "width": 960, "height": 528, "rgb_file": "frame-000.ppm",
                      "rgb_sha256": "a" * 64, "pts_seconds": 1.001}

    def manifest(self, frames):
        path = self.root / "input.json"
        path.write_text(json.dumps({"frames": frames}))
        return path

    def test_explicit_hash_accepts_small_native_batch_and_rejects_mutation(self):
        path = self.manifest([self.frame])
        digest = sha256(path)
        self.assertEqual(validated_inputs(path, digest)["frames"][0]["height"], 528)
        path.write_text(path.read_text() + " ")
        with self.assertRaises(ValueError):
            validated_inputs(path, digest)

    def test_reject_duplicate_frames_unsafe_paths_and_geometry(self):
        for frames in [[self.frame] * 2, [dict(self.frame, height=540)],
                       [dict(self.frame, rgb_file="../frame-000.ppm")], [self.frame] * 4]:
            path = self.manifest(frames)
            with self.assertRaises(ValueError):
                validated_inputs(path, sha256(path))

    def test_report_preserves_progress_when_next_write_is_invalid(self):
        path = self.root / "report.json"
        good = {"status": "started", "results": [{"count": 1.0}]}
        write_report_atomic(path, good)
        with self.assertRaises(ValueError):
            write_report_atomic(path, {"count": float("nan")})
        self.assertEqual(json.loads(path.read_text()), good)
        self.assertFalse(list(Path(os.environ["TMPDIR"]).glob("dronecrowd-report-*.json")))


if __name__ == "__main__":
    unittest.main()

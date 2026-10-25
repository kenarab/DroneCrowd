"""Aggregation checks: no double counting, spatial offsets and model units."""
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_crowd_candidates import clip_windows, streaming_clip, streaming_mpcount


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


if __name__ == "__main__":
    unittest.main()

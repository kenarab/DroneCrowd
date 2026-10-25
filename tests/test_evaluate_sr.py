"""SR spatial assembly and count-unit checks independent of trained outputs."""
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_sr import assemble, regional_sum, windows


class SRGeometryTests(unittest.TestCase):
    def test_coverage_alignment_and_bounded_context(self):
        for width, height in [(32, 32), (640, 528), (3840, 2160)]:
            coverage = np.zeros((height // 8, width // 8), np.uint8)
            for core, context in windows(width, height):
                x, y, right, bottom = core
                left, top, end_x, end_y = context
                self.assertTrue(all(v % 16 == 0 for v in (*core, *context)))
                self.assertLessEqual(end_x - left, 576)
                self.assertLessEqual(end_y - top, 576)
                coverage[y // 8:bottom // 8, x // 8:right // 8] += 1
            np.testing.assert_array_equal(coverage, 1)

    def test_spatial_offsets_preserved_and_sum_not_scaled(self):
        y, x = np.indices((528, 640))
        rgb = np.stack([x % 251, y % 251, (x + y) % 251], axis=-1).astype(np.uint8)

        def predict(crop):
            return (crop[::8, ::8, 0].astype(np.float32) + crop[::8, ::8, 1] * np.float32(256))

        density, tiles = assemble(rgb, predict, time.monotonic() + 5)
        np.testing.assert_array_equal(density, predict(rgb))
        self.assertEqual(tiles, 9)
        expected = predict(rgb)[16:32, 16:48].sum(dtype=np.float64)
        self.assertEqual(regional_sum(density, [128, 128, 384, 256]), expected)
        self.assertEqual(regional_sum(np.ones((8, 8), np.float32), [0, 0, 64, 64]), 64)

    def test_reject_invalid_geometry_and_roi(self):
        for dimensions in [(31, 32), (64, 65), (3856, 32)]:
            with self.assertRaises(ValueError):
                list(windows(*dimensions))
        for box in [[0, 0, 9, 16], [0, 0, 80, 64], [16, 0, 8, 16]]:
            with self.assertRaises(ValueError):
                regional_sum(np.zeros((8, 8), np.float32), box)

    def test_reject_nonfinite_negative_and_timeout(self):
        rgb = np.zeros((32, 32, 3), np.uint8)
        for value in [float("nan"), -1]:
            with self.assertRaises(ValueError):
                assemble(rgb, lambda _: np.full((4, 4), value, np.float32), time.monotonic() + 5)
        with self.assertRaises(TimeoutError):
            assemble(rgb, lambda _: np.ones((4, 4), np.float32), 0)


if __name__ == "__main__":
    unittest.main()

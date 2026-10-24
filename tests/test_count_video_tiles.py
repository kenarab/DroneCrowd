"""Coverage and density mass for tiled inference, without real-video fake counts."""
import sys
import time
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from count_video_tiles import assemble_density, quadrant_count, tile_windows


class TileCountingTests(unittest.TestCase):
    def test_native_and_partial_edges_cover_each_output_pixel_once(self):
        for width, height in [(3840, 2160), (960, 540), (1920, 1080), (32, 32), (644, 260)]:
            coverage = np.zeros((height // 2, width // 2), dtype=np.uint8)
            for core, context in tile_windows(width, height):
                x, y, right, bottom = core
                left, top, end, base = context
                self.assertTrue(all(value % 2 == 0 for value in core + context))
                self.assertLessEqual(end-left, 512)
                self.assertLessEqual(base-top, 320)
                self.assertGreaterEqual(end-left, 32)
                self.assertGreaterEqual(base-top, 32)
                coverage[y//2:bottom//2, x//2:right//2] += 1
            self.assertTrue(np.all(coverage == 1))

    def test_context_is_removed_without_shifting_or_duplicating_density(self):
        rgb = np.random.default_rng(9).integers(0, 256, (540, 960, 3), dtype=np.uint8)
        density, number = assemble_density(rgb, lambda crop: crop[::2, ::2, 0].astype(np.float32), time.monotonic()+10)
        np.testing.assert_array_equal(density, rgb[::2, ::2, 0])
        self.assertEqual(number, 15)
        count, _ = quadrant_count(density)
        self.assertAlmostEqual(count, float(density.sum()) / 100, places=4)

    def test_clamp_is_global_quadrant_sum_and_timeout_stops_before_prediction(self):
        density = np.array([[300, -200, -20, -30], [20, 30, 0, 0]], dtype=np.float32)
        count, sums = quadrant_count(density)
        self.assertEqual(count, 1.5)
        self.assertEqual(sums, [100, -50, 50, 0])
        with self.assertRaises(TimeoutError):
            assemble_density(np.zeros((32, 32, 3), dtype=np.uint8), None, time.monotonic()-1)

    def test_oversized_geometry_and_invalid_model_output_are_rejected(self):
        for shape in [(4000, 2160), (3840, 2164), (33, 32), (16, 32)]:
            with self.assertRaises(ValueError):
                list(tile_windows(*shape))
        for output in [np.zeros((1, 1), dtype=np.float32), np.full((16, 16), np.nan, dtype=np.float32)]:
            with self.assertRaises(ValueError):
                assemble_density(np.zeros((32, 32, 3), dtype=np.uint8), lambda crop: output, time.monotonic()+10)


if __name__ == "__main__":
    unittest.main()

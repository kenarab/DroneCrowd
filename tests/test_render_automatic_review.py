"""CPU-only checks for the existing density-peak postprocess and ROI join."""
import ast
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_automatic_review import calc_denpt, candidates_in_roi


class AutomaticReviewTests(unittest.TestCase):
    def test_matches_historical_postprocess_on_edges_plateaus_and_noise(self):
        source = Path(__file__).resolve().parents[1] / "STNNet/mytest.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "calc_denpt")
        namespace = {"np": np, "filters": ndimage, "ndimage": ndimage}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        density = np.random.default_rng(20261025).normal(0, .12, (40, 48)).astype(np.float32)
        density[0, 0], density[-1, -1] = 2., 3.
        density[12:15, 14:17] = 1.5
        x, y, peaks = namespace["calc_denpt"](density.copy()/100, 8, 2.0)
        np.testing.assert_array_equal(np.asarray(calc_denpt(density)), np.column_stack((x, y, peaks)))

    def test_calc_denpt_isolated_peak_uses_density_scale_and_stride_two(self):
        density = np.zeros((24, 24), dtype=np.float32)
        density[8, 9] = 100
        points = calc_denpt(density)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0][:2], (18.0, 16.0))
        self.assertEqual(points[0][2], 1.0)

    def test_flat_map_and_existing_diff_threshold_produce_no_candidates(self):
        self.assertEqual(calc_denpt(np.zeros((24, 24), dtype=np.float32)), [])
        density = np.zeros((24, 24), dtype=np.float32)
        density[8, 9] = 0.1  # After /100 this is exactly 0.001, not greater.
        self.assertEqual(calc_denpt(density), [])

    def test_peaks_are_found_in_context_then_half_open_roi_filtered_and_shifted(self):
        density = np.zeros((24, 24), dtype=np.float32)
        density[8, 8] = 100  # source=(116,116), included
        density[8, 16] = 90  # source=(132,116), right edge excluded
        density[16, 8] = 80  # source=(116,132), bottom edge excluded
        candidates = candidates_in_roi(density, [100, 100, 148, 148], [100, 100, 132, 132])
        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0]["x"], candidates[0]["y"]), (16.0, 16.0))
        self.assertNotIn("score", candidates[0])
        self.assertIn("peak_height", candidates[0])

if __name__ == "__main__":
    unittest.main()

"""Metric signs and denominators for the frozen annotated subset."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_calibration import error_metrics, historical_density


class CalibrationMetricTests(unittest.TestCase):
    def test_four_crop_order_assembly_and_heads_are_separate(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:32, :32] = 1
        rgb[:32, 32:] = 2
        rgb[32:, :32] = 3
        rgb[32:, 32:] = 4
        captured = []

        def fixture_model(inputs):
            tensor = inputs[0]
            captured.append(round(float((tensor[0, 0, 0, 0] * .229 + .485) * 255)))
            return [[torch.full((1, 1, 32//stride, 32//stride), float(len(captured)*head))]
                    for head, stride in enumerate((2, 4, 8), start=1)]

        density, counts = historical_density(rgb, fixture_model, device="cpu")
        self.assertEqual(captured, [1, 2, 3, 4])
        np.testing.assert_array_equal(density, np.repeat(np.repeat(np.array([[1, 2], [3, 4]]), 16, 0), 16, 1))
        for actual, expected in zip(counts, (25.6, 12.8, 4.8)):
            self.assertAlmostEqual(actual[0], expected)

    def test_signed_bias_absolute_error_and_rmse_use_same_denominator(self):
        metrics = error_metrics([{"count": 8, "gt_count": 10}, {"count": 14, "gt_count": 10}])
        self.assertEqual(metrics["images"], 2)
        self.assertEqual(metrics["mae"], 3)
        self.assertEqual(metrics["bias"], 1)
        self.assertAlmostEqual(metrics["rmse"], 10**.5)
        self.assertEqual(metrics["gt_total"], 20)
        self.assertEqual(metrics["prediction_total"], 22)
        for rows in ([], [{"count": float("nan"), "gt_count": 10}]):
            with self.assertRaises(ValueError):
                error_metrics(rows)


if __name__ == "__main__":
    unittest.main()

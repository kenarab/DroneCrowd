"""Pure checks for the bounded replay-control and initial-gradient summaries."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_domain_gradients import (measure_gradient, pairwise_source_products,
                                       gradient_cohorts, schedule_frequencies, weighted_aggregates)
from train_domain_pilot import require_training_count_gate, training_count_gate


class TrainingControlTests(unittest.TestCase):
    def test_mixed_recipe_frequencies_match_150_26_24(self):
        old, new = schedule_frequencies(16, 4)
        self.assertEqual(sum(old), 150)
        self.assertEqual(old, [10]*6 + [9]*10)
        self.assertEqual(new, [13, 13, 12, 12])
        rows = [{"supervision": "historical_full_crop", "frequency": f,
                 "gradient": [1., 0.]} for f in old]
        rows += [{"supervision": label, "frequency": f, "gradient": [0., 1.]}
                 for label, f in zip(("partial_positive", "partial_positive", "negative", "negative"), new)]
        summaries = weighted_aggregates(rows)
        self.assertEqual({key: value["steps"] for key, value in summaries.items()},
                         {"historical_full_crop": 150, "partial_positive": 26, "negative": 24})

    def test_pair_products_are_grouped_by_source_and_count_combinations(self):
        rows = [{"source_group": "a", "gradient": [1., 0.]},
                {"source_group": "a", "gradient": [0., 1.]},
                {"source_group": "b", "gradient": [-1., 0.]}]
        products = pairwise_source_products(rows)
        aa = next(row for row in products if row["source_a"] == "a" and row["source_b"] == "a")
        ab = next(row for row in products if row["source_a"] == "a" and row["source_b"] == "b")
        self.assertEqual(aa["pairs"], 4)
        self.assertEqual(ab["pairs"], 2)
        self.assertAlmostEqual(ab["mean_cosine"], -0.5)

    def test_train16_gate_accepts_band_and_rejects_inflation_or_collapse(self):
        report = training_count_gate([1.]*16, [0.5]*16)
        self.assertEqual(report["updated_to_original_ratio"], 0.5)
        self.assertEqual(len(report["original_counts"]), 16)
        self.assertEqual(len(report["updated_counts"]), 16)
        rejected = training_count_gate([1.]*16, [1.51]*16)
        self.assertFalse(rejected["accepted"])
        with self.assertRaisesRegex(ValueError, "gate failed"):
            require_training_count_gate(rejected)
        with self.assertRaisesRegex(ValueError, "gate failed"):
            require_training_count_gate(training_count_gate([1.]*16, [0.49]*16))

    def test_gradient_measurement_reads_rgb_relative_to_manifest_path(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.den_output_layer1 = torch.nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, images):
                pooled = torch.nn.functional.avg_pool2d(images[0], 2)
                return ([self.den_output_layer1(pooled)],)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ppm = root/"frame-000.ppm"
            ppm.write_bytes(b"P6\n32 32\n255\n" + bytes([128, 96, 64]) * (32*32))
            entry = {"rgb_file": ppm.name, "width": 32, "height": 32}
            target = np.full((16, 16), 0.01, dtype=np.float32)
            mask = np.ones((16, 16), dtype=bool)
            cohorts = gradient_cohorts([entry], [entry], root/"manifest.json",
                                       root/"controls.json", [1], [1])
            entries, manifest_path, _, is_new = cohorts[0]
            self.assertFalse(is_new)
            result = measure_gradient(TinyModel(), entries[0], manifest_path,
                                      target, mask, False, "cpu")
        self.assertEqual(len(result["gradient"]), 4)
        self.assertLessEqual(result["norm_after_clip"], 1.000001)
        self.assertEqual(sum(result["signs"].values()), 4)


if __name__ == "__main__":
    unittest.main()

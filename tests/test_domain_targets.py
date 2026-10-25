"""Scientific contracts: count units, ambiguous labels and one-sided gradients."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from domain_targets import density_loss, normalized_density, supervision_mask


class TargetTests(unittest.TestCase):
    def test_count_conservation_small_counts_edges_and_collisions(self):
        points = [[0, 0], [95.9, 95.9], [48, 48], [48, 48]]
        for count in range(5):
            target = normalized_density(points[:count], 96, 96)
            self.assertAlmostEqual(float(target.sum(dtype=np.float64)), count, places=6)
            self.assertTrue((target >= 0).all())

    def test_ambiguous_heads_and_crop_edges_are_unsupervised(self):
        _, mask = supervision_mask([[100, 100]], [[128, 128]], 256, 256, "exhaustive")
        self.assertFalse(mask[64, 64])
        self.assertFalse(mask[64, 72])
        self.assertTrue(mask[64, 73])
        self.assertFalse(mask[15].any())
        self.assertTrue(mask[16, 20])

    def test_partial_labels_do_not_create_background_supervision(self):
        target, mask = supervision_mask([[128, 128]], [], 256, 256, "partial_positive")
        self.assertFalse(mask[20, 20])
        self.assertTrue(mask[64, 64])
        self.assertTrue((target[mask] > 0).all())
        with self.assertRaises(ValueError):
            supervision_mask([], [], 256, 256, "partial_positive")

    def test_nonfinite_and_negative_label_misuse_fail(self):
        for points in ([[float("nan"), 4]], [[256, 4]], [[-1, 4]]):
            with self.assertRaises(ValueError):
                normalized_density(points, 256, 256)
        with self.assertRaises(ValueError):
            supervision_mask([[50, 50]], [], 256, 256, "negative")


class GradientTests(unittest.TestCase):
    def test_partial_loss_never_pushes_additional_people_down(self):
        import torch
        prediction = torch.tensor([[0., 3., 20.]], requires_grad=True)
        target = torch.tensor([[.01, .01, 0.]])
        mask = torch.tensor([[True, True, False]])
        density_loss(prediction, target, mask, partial=True).backward()
        self.assertLess(float(prediction.grad[0, 0]), 0)
        self.assertEqual(float(prediction.grad[0, 1]), 0)
        self.assertEqual(float(prediction.grad[0, 2]), 0)

    def test_full_supervision_has_correct_units_and_mask(self):
        import torch
        prediction = torch.tensor([[0., 3., 20.]], requires_grad=True)
        target = torch.tensor([[.01, .01, 0.]])
        mask = torch.tensor([[True, True, False]])
        loss = density_loss(prediction, target, mask)
        self.assertAlmostEqual(float(loss.detach()), 2.5)
        loss.backward()
        self.assertEqual(prediction.grad.tolist(), [[-1., 2., 0.]])


if __name__ == "__main__":
    unittest.main()

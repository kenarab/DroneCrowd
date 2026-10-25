"""Ensure pure-density optimizer steps change only explicitly selected weights."""
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from count_video import STNNet
from domain_targets import density_loss
from train_domain_pilot import collapse_gate, configure_trainable, new_target, state_hashes, verify_updates


class PilotTests(unittest.TestCase):
    def test_real_pure_density_step_changes_head_and_preserves_backbone(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        model = STNNet(use_loc=False, use_trk=False, pretrained_backbone=False)
        names = configure_trainable(model, "head")
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 129)
        before = state_hashes(model)
        prediction = model([torch.ones(1, 3, 32, 32)])[0][0][0, 0]
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        loss = density_loss(prediction, torch.ones_like(prediction)*.01,
                            torch.ones_like(prediction, dtype=torch.bool))
        loss.backward()
        optimizer.step()
        changed = verify_updates(before, state_hashes(model), names)
        self.assertIn("den_output_layer1.bias", changed)
        configure_trainable(model, "conv4-head")
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 147713)

    def test_context_pixels_outside_partial_roi_never_become_negatives(self):
        entry = {"width": 512, "height": 512, "roi_context_xyxy": [128, 128, 384, 384],
                 "points_confirmed_xy": [[128, 128]], "points_uncertain_xy": [], "supervision": "partial_positive"}
        target, mask = new_target(entry)
        self.assertAlmostEqual(float(target.sum(dtype=np.float64)), 1., places=6)
        self.assertFalse(mask[:64].any())
        self.assertFalse(mask[:, :64].any())
        self.assertTrue(mask[128, 128])

    def test_collapse_guard_uses_positive_targets_and_rejects_zero_counts(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, images):
                return ([torch.full((1, 1, 16, 16), self.value)],)

        class Cache:
            def get(self, entry):
                target = np.full((16, 16), 0 if entry["name"] == "negative" else .01, dtype=np.float32)
                return target, np.ones(target.shape, dtype=bool)

        entries = [{"name": n, "width": 32, "height": 32, "rgb_file": "unused.ppm"}
                   for n in ("negative", "one", "two", "three")]
        with patch("train_domain_pilot.read_rgb", return_value=np.zeros((32, 32, 3), dtype=np.uint8)):
            with self.assertRaisesRegex(ValueError, "collapse"):
                collapse_gate(ConstantModel(-.01), entries, Path("manifest.json"), Cache(), time.monotonic()+5, "cpu")
            rows = collapse_gate(ConstantModel(.01), entries, Path("manifest.json"), Cache(), time.monotonic()+5, "cpu")
        self.assertEqual([r["name"] for r in rows], ["one", "two", "three"])
        self.assertTrue(all(r["count"] > 0 for r in rows))


if __name__ == "__main__":
    unittest.main()

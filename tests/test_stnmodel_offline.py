"""Smoke del núcleo STNNet: tensores sintéticos, sin checkpoints ni descargas."""
import builtins
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "STNNet"))
from stnmodel import STNNet

torch.set_num_threads(1)


class OfflineModelTests(unittest.TestCase):
    def devices(self):
        devices = ["cpu"]
        if os.environ.get("DRONECROWD_TEST_CUDA") == "1":
            self.assertTrue(torch.cuda.is_available(), "CUDA requerida por el diagnóstico")
            devices.append("cuda")
        return devices

    def test_density_forward_needs_no_tracking_or_backbone_download(self):
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torchvision", "spatial_correlation_sampler", "pointconv"):
                raise AssertionError("Dependencia opcional utilizada en conteo: " + name)
            return original_import(name, *args, **kwargs)

        for device in self.devices():
            with self.subTest(device=device), patch("builtins.__import__", restricted_import):
                model = STNNet(pretrained_backbone=False).to(device).eval()
                with torch.no_grad():
                    outputs = model([torch.zeros(1, 3, 32, 32, device=device) for _ in range(2)])
                self.assertEqual(len(outputs), 14)
                for maps, side in zip(outputs[:3], (16, 8, 4)):
                    self.assertEqual(len(maps), 2)
                    for density in maps:
                        self.assertEqual(tuple(density.shape), (1, 1, side, side))
                        self.assertTrue(torch.isfinite(density).all().item())
                self.assertTrue(all(not output for output in outputs[3:]))

    def test_localization_forward_preserves_all_heads_without_tracking(self):
        for device in self.devices():
            with self.subTest(device=device):
                model = STNNet(use_loc=True, pretrained_backbone=False).to(device).eval()
                with torch.no_grad():
                    outputs = model([torch.zeros(1, 3, 32, 32, device=device) for _ in range(2)])
                for maps, side in zip(outputs[3:11], (16, 8, 4, 16, 16, 8, 4, 16)):
                    self.assertEqual(len(maps), 2)
                    for output in maps:
                        self.assertEqual(tuple(output.shape), (1, 2, side, side))
                        self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue(all(not output for output in outputs[11:]))


if __name__ == "__main__":
    unittest.main()

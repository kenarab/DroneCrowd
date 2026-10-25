"""Check historical grid, edge loss, collisions, and fixed-sigma sensitivity."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_target_kernels import (HEIGHT, WIDTH, adaptive_sigmas, add_impulse,
                                     density_map, isolated_peak_ratio, occupied_cells,
                                     reconstruct_density)
from audit_hdf5 import reconstruct_density as audit_reconstruct_density


class TargetKernelTests(unittest.TestCase):
    def test_edge_kernel_keeps_historical_lost_mass(self):
        edge = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        retained = add_impulse(edge, 0, 0, 2.0)
        center = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        centered = add_impulse(center, 100, 100, 2.0)
        self.assertLess(retained, 0.5)
        self.assertAlmostEqual(centered, 1.0, places=6)
        self.assertAlmostEqual(float(edge.sum()), retained, places=6)

    def test_same_truncated_cell_collides_as_one_boolean_impulse(self):
        occupied, coordinates, collisions = occupied_cells([[2.1, 2.1], [2.9, 2.9]])
        self.assertEqual(int(occupied.sum()), 1)
        self.assertEqual(coordinates.tolist(), [[1, 1]])
        self.assertEqual(collisions, 1)
        self.assertEqual(len(adaptive_sigmas(coordinates)), 1)

    def test_fixed_sigma_changes_spatial_kernel_and_isolated_peak(self):
        coordinates = np.asarray([[10, 10], [11, 10], [10, 11], [11, 11]])
        sigmas = adaptive_sigmas(coordinates)
        adaptive, _ = density_map(coordinates, sigmas)
        fixed, _ = density_map(coordinates, np.full(4, 2.0))
        self.assertFalse(np.allclose(adaptive, fixed))
        ratios = isolated_peak_ratio(sigmas)
        self.assertTrue(all(ratio > 1 for ratio in ratios))

    def test_adaptive_map_is_bitwise_equal_to_audit_oracle(self):
        points = np.asarray([[40, 42], [72, 46], [48, 80], [90, 94], [120, 60]], dtype=np.float64)
        _, _, _, _, adaptive, _ = reconstruct_density(points)
        oracle = audit_reconstruct_density(points, height=HEIGHT, width=WIDTH)
        self.assertTrue(np.array_equal(adaptive, oracle))


if __name__ == "__main__":
    unittest.main()

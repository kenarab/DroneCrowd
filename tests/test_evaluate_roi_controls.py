"""ROI spatial alignment and density mass; no provisional labels as true GT."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_roi_controls import roi_count


class ROIIntegrationTests(unittest.TestCase):
    def test_scale_changes_grid_and_does_not_multiply_person_count(self):
        self.assertEqual(roi_count(np.full((8,8),100,dtype=np.float32),[4,4,12,12],1),(16,16))
        self.assertEqual(roi_count(np.full((16,16),25,dtype=np.float32),[4,4,12,12],2),(16,16))
        self.assertEqual(roi_count(np.full((8,8),-100,dtype=np.float32),[4,4,12,12],1),(0,-16))

    def test_outside_empty_unaligned_and_nonfinite_regions_are_rejected(self):
        density=np.ones((8,8),dtype=np.float32)
        for box in ([-4,0,4,4],[0,0,20,4],[4,4,4,8],[2,0,4,4]):
            with self.assertRaises(ValueError):
                roi_count(density,box,1)
        with self.assertRaises(ValueError):
            roi_count(np.full((8,8),np.nan,dtype=np.float32),[0,0,4,4],1)


if __name__=="__main__":
    unittest.main()

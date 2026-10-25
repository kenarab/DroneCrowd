"""Original MAT point/identity schema, without map generation or GPU."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_calibration import ground_truth


class CalibrationPreparationTests(unittest.TestCase):
    def test_count_is_rows_and_ids_remain_metadata(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="dronecrowd-gt-test-") as directory:
            path = Path(directory)/"points.mat"
            points = np.array([[1.5, 2.5, 200], [30, 31, 500]])
            cell = np.empty((1, 1), dtype=object)
            cell[0, 0] = {"location": points, "number": np.array([[2]])}
            savemat(path, {"image_info": cell})
            np.testing.assert_array_equal(ground_truth(path, 32, 32), points)
            for invalid_points, number in [(points, 2.5), (points, 3),
                                           (np.array([[1, 2, 200], [3, 4, 200]]), 2),
                                           (np.array([[32, 2, 1]]), 1),
                                           (np.array([[float("nan"), 2, 1]]), 1)]:
                cell[0, 0] = {"location": invalid_points, "number": np.array([[number]])}
                savemat(path, {"image_info": cell})
                with self.assertRaises(ValueError):
                    ground_truth(path, 32, 32)


if __name__ == "__main__":
    unittest.main()

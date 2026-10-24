"""Independent dense references and safety checks for the read-only audit."""
import ctypes as ct
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage, spatial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_hdf5 import HDF5Reader, reconstruct_density, reconstruct_localization
from regenerate_hdf5 import write_new_file


class ReconstructionTests(unittest.TestCase):
    def test_density_matches_full_image_gaussian_including_edges_and_overlaps(self):
        points = np.array([[0, 0, 1], [6, 4, 2], [8, 4, 3], [38, 26, 4],
                           [12, 14, 5], [13, 15, 6], [20, 8, 7]], dtype=float)
        occupied = np.zeros((14, 20), dtype=np.float64)
        for x, y, _ in points:
            occupied[int(y / 2), int(x / 2)] = 1
        ys, xs = np.nonzero(occupied)
        coordinates = np.column_stack((xs, ys))
        distances, _ = spatial.KDTree(coordinates.copy(), leafsize=2048).query(coordinates, k=4)
        expected = np.zeros((14, 20), dtype=np.float32)
        for index, (x, y) in enumerate(coordinates):
            impulse = np.zeros(expected.shape, dtype=np.float32)
            impulse[y, x] = 1
            sigma = (distances[index, 1] + distances[index, 2] + distances[index, 3]) * 0.1
            expected += ndimage.gaussian_filter(impulse, sigma, mode="constant")
        np.testing.assert_array_equal(reconstruct_density(points, 14, 20), expected)

    def test_empty_and_single_person_density(self):
        np.testing.assert_array_equal(reconstruct_density(np.empty((0, 3)), 14, 20), np.zeros((14, 20)))
        expected = np.zeros((14, 20), dtype=np.float32)
        expected[0, 0] = 1
        expected = ndimage.gaussian_filter(expected, (14 + 20) / 8, mode="constant")
        np.testing.assert_array_equal(reconstruct_density(np.array([[0, 0, 1]]), 14, 20), expected)

    def test_localization_matches_original_nested_pixel_loops(self):
        points = np.array([[0, 0, 1], [6, 4, 2], [8, 4, 3], [38, 30, 4]], dtype=float)
        expected = {"identity": np.zeros((16, 20), dtype=np.int16)}
        for level in range(3):
            height, width = 16 // 2**level, 20 // 2**level
            expected[f"offset{level + 1}"] = np.zeros((height, width, 2), dtype=np.float64)
        for x_value, y_value, person in points:
            for level in range(3):
                offset = expected[f"offset{level + 1}"]
                height, width = offset.shape[:2]
                y, x = int(y_value / 2**(level + 1)), int(x_value / 2**(level + 1))
                if x < width and y < height:
                    if level == 0:
                        expected["identity"][y, x] = person
                    radius = int(5 / (level + 1.0))
                    for column in range(max(0, x - radius), min(width - 1, x + radius) + 1):
                        for row in range(max(0, y - radius), min(height - 1, y + radius) + 1):
                            offset[row, column, 0], offset[row, column, 1] = y - row, x - column
        actual = reconstruct_localization(points, 16, 20)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_negative_coordinates_and_historical_infinite_sigma_are_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_density(np.array([[-4, -4, 1]]), 14, 20)
        with self.assertRaises(ValueError):
            reconstruct_density(np.array([[0, 0, 1], [4, 4, 2]]), 14, 20)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dronecrowd-hdf5-test-", dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.h5"
        self.reader = HDF5Reader()
        self.lib = self.reader.lib
        signatures = {
            "H5Fcreate": (ct.c_int64, [ct.c_char_p, ct.c_uint, ct.c_int64, ct.c_int64]),
            "H5Screate_simple": (ct.c_int64, [ct.c_int, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)]),
            "H5Dcreate2": (ct.c_int64, [ct.c_int64, ct.c_char_p] + [ct.c_int64] * 5),
            "H5Dwrite": (ct.c_int, [ct.c_int64] * 5 + [ct.c_void_p]),
            "H5Lcreate_soft": (ct.c_int, [ct.c_char_p, ct.c_int64, ct.c_char_p, ct.c_int64, ct.c_int64]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, arguments

    def create(self, array, softlink=False):
        file_id = self.reader.checked(self.lib.H5Fcreate(os.fsencode(self.path), 2, 0, 0))
        datatype = ct.c_int64.in_dll(self.lib, "H5T_NATIVE_FLOAT_g").value
        dimensions = (ct.c_uint64 * 2)(*array.shape)
        space = self.reader.checked(self.lib.H5Screate_simple(2, dimensions, None))
        dataset = self.reader.checked(self.lib.H5Dcreate2(file_id, b"density", datatype, space, 0, 0, 0))
        try:
            self.reader.checked(self.lib.H5Dwrite(dataset, datatype, 0, 0, 0, array.ctypes.data))
            if softlink:
                self.reader.checked(self.lib.H5Lcreate_soft(b"/density", file_id, b"copy", 0, 0))
        finally:
            self.lib.H5Dclose(dataset)
            self.lib.H5Sclose(space)
            self.lib.H5Fclose(file_id)

    def test_reads_values_without_modifying_file(self):
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.create(expected)
        before = self.path.read_bytes()
        metadata, arrays = self.reader.read(self.path, True)
        self.assertEqual(metadata["density"]["shape"], (3, 4))
        np.testing.assert_array_equal(arrays["density"], expected)
        self.assertEqual(before, self.path.read_bytes())

    def test_nonfinite_payload_is_rejected(self):
        self.create(np.full((3, 4), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.reader.read(self.path, True)

    def test_soft_link_is_rejected(self):
        self.create(np.zeros((3, 4), dtype=np.float32), softlink=True)
        with self.assertRaisesRegex(ValueError, "link"):
            self.reader.read(self.path, True)

    def test_writer_roundtrips_all_map_datatypes(self):
        arrays = {"density": np.arange(12, dtype=np.float32).reshape(3, 4),
                  "identity": np.arange(12, dtype=np.int16).reshape(3, 4),
                  "offset1": np.arange(24, dtype=np.float64).reshape(3, 4, 2)}
        write_new_file(self.path, arrays, self.reader)
        _, actual = self.reader.read(self.path, True)
        for name, expected in arrays.items():
            self.assertEqual(actual[name].dtype, expected.dtype)
            np.testing.assert_array_equal(actual[name], expected)

    def test_writer_refuses_existing_file(self):
        original = b"previous scientific data"
        self.path.write_bytes(original)
        with self.assertRaises(FileExistsError):
            write_new_file(self.path, {"density": np.zeros((3, 4), dtype=np.float32)}, self.reader)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

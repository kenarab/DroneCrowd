import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from domain_annotations import mat_payload, validate_manifest


def fixture():
    return {"schema": "dronecrowd-head-points-v1", "annotation_provenance": "test provisional",
            "entries": [{"name": "example", "split": "train", "source_group": "photo-a",
                "source_sha256": "a"*64, "box_xyxy": [0, 0, 256, 256],
                "points_confirmed_xy": [[100, 120], [160, 180]], "points_uncertain_xy": [],
                "supervision": "exhaustive", "review_status": "agent_provisional"}]}


class AnnotationTests(unittest.TestCase):
    def test_mat_roundtrip_matches_original_nested_schema_and_local_ids(self):
        entry = fixture()["entries"][0]
        buffer = io.BytesIO()
        savemat(buffer, mat_payload(entry))
        buffer.seek(0)
        fields = loadmat(buffer)["image_info"][0, 0][0, 0]
        np.testing.assert_array_equal(fields["location"], [[100, 120, 1], [160, 180, 2]])
        self.assertEqual(fields["number"].item(), 2)

    def test_same_source_or_video_cannot_cross_splits(self):
        for field in ("source_group", "source_sha256"):
            data = fixture()
            other = copy.deepcopy(data["entries"][0])
            other.update(name="heldout", split="test", source_group="photo-b", source_sha256="b"*64)
            other[field] = data["entries"][0][field]
            data["entries"].append(other)
            with self.assertRaisesRegex(ValueError, "crossed"):
                validate_manifest(data)

    def test_exact_duplicate_click_rejected(self):
        data = fixture()
        data["entries"][0]["points_uncertain_xy"] = [[100, 120]]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(data)


if __name__ == "__main__":
    unittest.main()

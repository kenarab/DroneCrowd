"""Numerical recipe and restricted-checkpoint guards; no real-video inference."""
import os
import pickle
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from count_video import STNNet, count_from_maps, load_density_checkpoint, normalized_rgb, quadrants, read_rgb, restricted_checkpoint, validate_frames
from sample_video import sha256

torch.set_num_threads(1)


class VideoCountingTests(unittest.TestCase):
    def test_localization_checkpoint_requires_explicit_matching_architecture(self):
        state=STNNet(use_loc=True,pretrained_backbone=False).state_dict()
        with tempfile.TemporaryDirectory(prefix="dronecrowd-loc-checkpoint-",dir=os.environ["TMPDIR"]) as directory:
            path=Path(directory)/"mock-localization.pt"
            path.write_bytes(b"mock-localization-fixture")
            with patch("count_video.torch.load",return_value={"state_dict":state}):
                with self.assertRaisesRegex(ValueError,"keys differ"):
                    load_density_checkpoint(path,sha256(path))
                model=load_density_checkpoint(path,sha256(path),use_loc=True)
            self.assertTrue(model.loc)
            self.assertFalse(model.trk)
            for name,tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor,state[name]))

    def test_legacy_numpy_metadata_is_hash_scoped_and_other_globals_stay_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dronecrowd-numpy-metadata-", dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "metadata.pt"
            buffer = io.BytesIO()
            torch.save({"best_prec1": np.float64(16.8835)}, buffer, pickle_protocol=2)
            # Reproduce the old module spelling observed in the actual pickle.
            with zipfile.ZipFile(buffer) as source, zipfile.ZipFile(path, "w") as target:
                for entry in source.infolist():
                    data = source.read(entry)
                    if entry.filename.endswith("data.pkl"):
                        data = data.replace(b"numpy._core.multiarray\n", b"numpy.core.multiarray\n")
                    target.writestr(entry, data)
            digest = sha256(path)
            before = torch.serialization.get_safe_globals().copy()
            with self.assertRaises(pickle.UnpicklingError):
                restricted_checkpoint(path, digest)
            with patch("count_video.NUMPY_METADATA_CHECKPOINTS", {digest}):
                self.assertEqual(restricted_checkpoint(path, digest)["best_prec1"], np.float64(16.8835))
            self.assertEqual(torch.serialization.get_safe_globals(), before)
            torch.save({"unrelated": Path("untrusted-path")}, path)
            with patch("count_video.NUMPY_METADATA_CHECKPOINTS", {sha256(path)}):
                with self.assertRaises(pickle.UnpicklingError):
                    restricted_checkpoint(path, sha256(path))
            self.assertEqual(torch.serialization.get_safe_globals(), before)

    def test_rgb_channel_order_and_imagenet_normalization(self):
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        actual = normalized_rgb(rgb)
        expected = torch.tensor([[(1 - .485) / .229, -.456 / .224, -.406 / .225]]).reshape(1, 3, 1, 1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float32)

    def test_quadrants_cover_every_pixel_once(self):
        rgb = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
        crops = quadrants(rgb)
        assembled = np.concatenate([np.concatenate(crops[:2], axis=1),
                                    np.concatenate(crops[2:], axis=1)], axis=0)
        np.testing.assert_array_equal(rgb, assembled)
        with self.assertRaises(ValueError):
            quadrants(np.zeros((33, 48, 3), dtype=np.uint8))

    def test_count_uses_crop_sums_clamped_after_summing_and_factor_one_hundred(self):
        maps = [np.array([[300, -200]], dtype=np.float32), np.array([[-20, -30]], dtype=np.float32),
                np.array([[20, 30]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32)]
        count, sums = count_from_maps(maps)
        self.assertEqual(sums, [100, -50, 50, 0])
        self.assertEqual(count, 1.5)
        maps[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            count_from_maps(maps)

    def test_checkpoint_hash_and_wrong_keys_are_rejected_with_restricted_load(self):
        with tempfile.TemporaryDirectory(prefix="dronecrowd-checkpoint-test-", dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "synthetic-wrong-keys.pt"
            torch.save({"state_dict": {"wrong": torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_density_checkpoint(path, "0" * 64)
            with patch("count_video.torch.load", wraps=torch.load) as loader:
                with self.assertRaisesRegex(ValueError, "keys differ"):
                    load_density_checkpoint(path, sha256(path))
                self.assertTrue(loader.call_args.kwargs["weights_only"])
                self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")

    def test_checkpoint_prefix_collision_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dronecrowd-checkpoint-test-", dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "synthetic-collision.pt"
            torch.save({"state_dict": {"same": torch.zeros(1), "module.same": torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, "collision"):
                load_density_checkpoint(path, sha256(path))

    def test_nonfinite_sum_and_oversized_input_are_rejected(self):
        with self.assertRaises(ValueError):
            count_from_maps([np.full((10, 10), 1e38, dtype=np.float32)] * 4)
        with self.assertRaises(ValueError):
            read_rgb("not-read", 100000, 100000)

    def test_complete_prefixed_state_loads_and_mismatched_tensor_is_rejected(self):
        # Mock only the serialized fixture, never run synthetic weights on a video.
        expected = STNNet(pretrained_backbone=False).state_dict()
        with tempfile.TemporaryDirectory(prefix="dronecrowd-checkpoint-test-", dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "mock-serialized-state.pt"
            path.write_bytes(b"unit-test-fixture")
            prefixed = {"module." + key: value for key, value in expected.items()}
            with patch("count_video.torch.load", return_value={"state_dict": prefixed}):
                loaded = load_density_checkpoint(path, sha256(path))
                for name, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]))
                self.assertFalse(loaded.training)
            del loaded
            first = next(iter(prefixed))
            for invalid in (torch.zeros(1), torch.full_like(prefixed[first], float("nan")), prefixed[first].double()):
                changed = dict(prefixed, **{first: invalid})
                with patch("count_video.torch.load", return_value={"state_dict": changed}):
                    with self.assertRaisesRegex(ValueError, "shape/dtype/finiteness"):
                        load_density_checkpoint(path, sha256(path))

    def test_manifest_caps_are_checked_before_any_frame_read(self):
        frame = {"index": 0, "width": 100000, "height": 100000, "pts_seconds": 1.0,
                 "rgb_file": "frame-000.ppm", "rgb_sha256": "0" * 64}
        with patch("pathlib.Path.lstat", side_effect=AssertionError("Must reject before file access")):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                validate_frames([frame], Path("unused"))
        frame.update(width=960, height=540, pts_seconds=float("nan"))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_frames([frame], Path("unused"))


if __name__ == "__main__":
    unittest.main()

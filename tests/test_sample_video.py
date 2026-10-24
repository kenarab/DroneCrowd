"""Real FFmpeg sampling and source-preservation checks on a tiny fixture."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sample_video import geometry, output_budget, sample, sha256


class VideoSampleTests(unittest.TestCase):
    def test_geometry_preserves_common_aspect_with_four_complete_quadrants(self):
        self.assertEqual(geometry(3840, 2160, 960), (960, 540))
        for width in (31, 961, 1924):
            with self.assertRaises(ValueError):
                geometry(3840, 2160, width)

    def test_frame_count_and_nonfinite_fractions_are_rejected_before_io(self):
        for fractions in ([], [0.1] * 31, [float("nan")], [1.0]):
            with self.assertRaises(ValueError):
                sample("missing-video", "unused-output", fractions=fractions)

    def test_png_budget_includes_an_uncompressed_rgb_sized_preview(self):
        self.assertLess(output_budget(960, 540, 30), 120 * 2**20)
        with self.assertRaises(ValueError):
            output_budget(1920, 1080, 30)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_real_decode_records_actual_timestamps_and_keeps_original(self):
        with tempfile.TemporaryDirectory(prefix="dronecrowd-video-test-", dir=os.environ["TMPDIR"]) as directory:
            root = Path(directory)
            source_dir = root / "originals"
            source_dir.mkdir()
            video = source_dir / "fixture.mkv"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
                            "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=10:duration=2",
                            "-c:v", "ffv1", "-threads", "1", str(video)], check=True, timeout=15)
            original = sha256(video)
            report = sample(video, root / "sample", width=64)
            self.assertEqual(report["source_sha256"], original)
            self.assertEqual(sha256(video), original)
            self.assertFalse(report["model_inference"])
            self.assertEqual(len(report["frames"]), 3)
            for frame, target in zip(report["frames"], (0.2, 1.0, 1.8)):
                self.assertAlmostEqual(frame["pts_seconds"], target, places=5)
                self.assertEqual(sha256(root / "sample" / frame["rgb_file"]), frame["rgb_sha256"])
                self.assertTrue((root / "sample" / frame["preview_file"]).is_file())
            native = sample(video, root / "native", width=32, fractions=(.5,), native=True)
            self.assertEqual((native["frames"][0]["width"], native["frames"][0]["height"]), (64, 36))
            self.assertEqual(native["frames"][0]["rgb_sha256"], report["frames"][1]["rgb_sha256"])
            self.assertEqual(native["sampling_transform"], "native RGB decode, no resize")
            with self.assertRaises(ValueError):
                sample(video, source_dir / "forbidden", width=64)


if __name__ == "__main__":
    unittest.main()

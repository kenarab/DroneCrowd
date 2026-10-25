"""Guard pending annotation language and prediction-to-ROI provenance joins."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from render_extended_review import evaluation_label, pending, predictions_for, render_roi, same_roi, summary
from sample_video import sha256


class ExtendedReviewTests(unittest.TestCase):
    def test_pending_regions_are_not_rendered_as_zero_references(self):
        pending_summary = summary({"points_empty": True, "points_confirmed_xy": [],
                                   "points_uncertain_xy": [], "reference_interval": None})
        self.assertEqual(pending_summary, {"status": "pending", "nominal": None, "uncertain": None})
        self.assertTrue(pending({
            "review_status": "pending_joint_review", "supervision": "unannotated",
            "points_confirmed_xy": [], "points_uncertain_xy": []}))
        marked = summary({"points_confirmed_xy": [[12, 14]], "points_uncertain_xy": [[20, 18]]})
        self.assertEqual(marked["nominal"], 2)
        self.assertEqual(marked["uncertain"], 1)

    def test_composite_png_preserves_native_roi_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"roi.png"
            scale = render_roi(Image.new("RGB", (24, 17), "gray"),
                               {"points_confirmed_xy": [], "points_uncertain_xy": []}, path)
            with Image.open(path) as result:
                self.assertEqual(result.size, (48, 17))
            self.assertEqual(scale, 1.0)

    def test_prediction_join_requires_name_source_hash_and_exact_box(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)/"frame.jpg"
            Image.new("RGB", (64, 64)).save(source)
            target = {"name": "roi-a", "source_image": str(source), "source_sha256": "a"*64,
                      "box_xyxy": [0, 0, 32, 32]}
            self.assertTrue(same_roi(target, dict(target)))
            self.assertFalse(same_roi(target, {**target, "box_xyxy": [0, 0, 36, 32]}))
            self.assertFalse(same_roi(target, {**target, "name": "other"}))

    def test_nonoriginal_checkpoints_are_never_labeled_as_original(self):
        label = evaluation_label({"checkpoint": "/some/mixto-conv4-head/model.pth",
                                  "checkpoint_sha256": "a"*64}, Path("evaluation.json"))
        self.assertIn("rechazado", label)
        original = evaluation_label({"checkpoint_sha256":
            "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d"}, Path("e.json"))
        self.assertEqual(original, "original")

    def test_evaluation_count_is_reused_only_after_control_hash_and_geometry_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source.jpg"
            Image.new("RGB", (64, 64), "navy").save(source)
            target = {"name": "roi-a", "source_image": str(source), "source_sha256": sha256(source),
                      "box_xyxy": [0, 0, 32, 32]}
            controls_path, eval_path = root/"controls.json", root/"evaluation.json"
            control = {**target, "index": 0}
            controls_path.write_text(json.dumps({"entries": [control]}))
            evaluation = {"status": "complete", "scale": 1, "variant": "density", "controls": str(controls_path),
                "controls_sha256": sha256(controls_path), "checkpoint": "original",
                "checkpoint_sha256": "be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d",
                "rows": [{"name": "roi-a", "index": 0, "count": 3.25}]}
            eval_path.write_text(json.dumps(evaluation))
            matched, mismatches, _ = predictions_for(eval_path, [target])
            self.assertEqual(matched["roi-a"]["count"], 3.25)
            self.assertEqual(mismatches, [])
            controls_path.write_text(json.dumps({"entries": [{**control, "box_xyxy": [0, 0, 36, 32]}]}))
            evaluation["controls_sha256"] = sha256(controls_path)
            eval_path.write_text(json.dumps(evaluation))
            matched, mismatches, _ = predictions_for(eval_path, [target])
            self.assertEqual(matched, {})
            self.assertEqual(mismatches, ["roi-a"])
            for invalid in ({"scale": 2}, {"variant": "localization"}):
                eval_path.write_text(json.dumps({**evaluation, **invalid}))
                with self.assertRaisesRegex(ValueError, "scale 1 density"):
                    predictions_for(eval_path, [target])


if __name__ == "__main__":
    unittest.main()

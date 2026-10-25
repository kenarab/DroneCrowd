import importlib.util
import errno
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TMP = Path("/home/alephbet/home.codex/tmp")
SPEC = importlib.util.spec_from_file_location(
    "prepare_long_video_jobs", ROOT / "scripts/prepare_long_video_jobs.py")
jobs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jobs)


class PrepareLongVideoJobsTests(unittest.TestCase):
    def test_sparse_times_batches_and_budget(self):
        cases = [("0265", 22.856167, 3840, 2160, 23, 12),
                 ("0269", 9.392717, 3840, 2160, 10, 5),
                 ("trim", 10.985, 960, 540, 11, 6)]
        all_batches = []
        for name, duration, width, height, frame_count, batch_count in cases:
            batches = jobs.plan_batches(name, duration, width, height,
                                        12 if name == "trim" else 0)
            self.assertEqual(len(jobs.integer_sample_times(duration)), frame_count)
            self.assertEqual(len(batches), batch_count)
            self.assertTrue(all(1 <= batch["frame_count"] <= 2 for batch in batches))
            self.assertTrue(all(batch["sampler_output_budget_estimate_bytes"]
                                <= jobs.MAX_BATCH_BYTES for batch in batches))
            all_batches.extend(batches)
        self.assertEqual(len(all_batches), 23)
        self.assertEqual(sum(batch["frame_count"] for batch in all_batches), 44)
        trim = next(batch for batch in all_batches if batch["video_id"] == "trim")
        self.assertEqual(trim["effective_dimensions"], [960, 528])
        self.assertEqual(trim["crop_xyxy"], [0, 0, 960, 528])

    def test_plan_has_69_local_jobs_and_no_false_enqueue(self):
        batches = jobs.plan_batches("fake", 3, 64, 64)
        source = {"video_id": "fake", "source_manifest": "x/input.json",
                  "source_manifest_sha256": "a", "source_sha256": "b",
                  "source_bytes": 1, "duration_seconds": 3, "width": 64,
                  "height": 64, "source_path_local": "/local/video",
                  "crop_last_rows": 0, "batches": batches}
        models = {name: {"path": f"weights/{name}", "sha256": name*64}
                  for name in ("sr", "clip-ebc", "mpcount")}
        code = {"runner.py": "1"*64}
        prereq = {"clip_ebc_source": {"path_local": "/offline/clip"}}
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            output = ROOT / "outputs" / Path(temporary).name
            plan = jobs.build_plan(output, [source], models, code, "revision", prereq)
        self.assertEqual(plan["estimated_jobs"], 6)
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertTrue(plan["not_submitted"])
        self.assertFalse(plan["executor_profile_registered"])
        self.assertTrue(all(job["job_id"] is None and not job["materialized"]
                            for job in plan["jobs"]))
        self.assertEqual(plan["preparation_state"], "planned")

    def test_idempotency_binds_inputs_checkpoint_and_runtime(self):
        code = {"a.py": "1"*64}
        key = jobs.idempotency_key("sr", "a"*64, "b"*64, code)
        self.assertEqual(key, jobs.idempotency_key("sr", "a"*64, "b"*64, code))
        self.assertNotEqual(key, jobs.idempotency_key("sr", "c"*64, "b"*64, code))
        self.assertNotEqual(key, jobs.idempotency_key("sr", "a"*64, "d"*64, code))
        self.assertNotEqual(key, jobs.idempotency_key("sr", "a"*64, "b"*64,
                                                      {"a.py": "2"*64}))
        with self.assertRaises(ValueError):
            jobs.idempotency_key("sr", None, "b"*64, code)

    def test_trim_crop_is_native_and_removes_exactly_twelve_rows(self):
        from PIL import Image
        from PIL import ImageDraw
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            directory = Path(temporary)
            ppm = directory / "frame.ppm"
            image = Image.new("RGB", (960, 540), (7, 11, 13))
            ImageDraw.Draw(image).rectangle((0, 528, 959, 539), fill=(240, 20, 30))
            image.save(ppm, format="PPM")
            frame = {"width": 960, "height": 540, "rgb_file": ppm.name}
            jobs._crop_trim_batch(directory, frame)
            self.assertEqual((frame["width"], frame["height"]), (960, 528))
            self.assertEqual(frame["cropped_bottom_rows"], 12)
            self.assertEqual(len(jobs._ppm_rgb(ppm, 960, 528)), 960*528*3)
            with Image.open(ppm) as image:
                self.assertEqual(image.size, (960, 528))
                self.assertEqual(image.getpixel((3, 0)), (7, 11, 13))
                self.assertEqual(image.getpixel((3, 527)), (7, 11, 13))

    def test_existing_batch_verification_checks_hash_and_source(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            directory = Path(temporary)
            ppm = directory / "frame.ppm"
            payload = b"\x01\x02\x03" * 4
            ppm.write_bytes(b"P6\n2 2\n255\n" + payload)
            frame = {"index": 0, "pts_seconds": 0.0, "requested_seconds": 0,
                     "width": 2, "height": 2, "rgb_file": ppm.name,
                     "rgb_sha256": jobs.sha256_file(ppm)}
            manifest = {"source_sha256": "source", "frames": [frame]}
            (directory / "input.json").write_text(json.dumps(manifest))
            batch = {"batch_id": "fake-000", "frame_times_seconds": [0],
                     "effective_dimensions": [2, 2], "cropped_bottom_rows": 0}
            source = {"source_sha256": "source"}
            verified = jobs._validate_batch(directory, batch, source)
            self.assertEqual(verified["pts_seconds"], [0.0])
            self.assertEqual(len(verified["input_sha256"]), 64)
            with self.assertRaises(ValueError):
                jobs._validate_batch(directory, batch, {"source_sha256": "other"})
            frame["pts_seconds"] = 0.1
            (directory / "input.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                jobs._validate_batch(directory, batch, source)
            frame["pts_seconds"] = 0.0
            (directory / "input.json").write_text(json.dumps(manifest))
            ppm.write_bytes(b"P6\n2 2\n255\n" + b"\x00"*12)
            with self.assertRaises(ValueError):
                jobs._validate_batch(directory, batch, source)

    def test_preflight_keeps_reserve_after_estimate_and_margin(self):
        required = 20*2**30 + int(4*2**30*1.30)
        with self.assertRaises(RuntimeError):
            jobs.preflight("unused", free_bytes=required-1, available_ram_bytes=3*2**30)
        report = jobs.preflight("unused", free_bytes=required, available_ram_bytes=3*2**30)
        self.assertEqual(report["minimum_free_disk_bytes"], required)
        self.assertEqual(report["minimum_retained_disk_bytes"], 20*2**30)

    def test_staged_interruption_resume_and_frozen_manifest_hash(self):
        from unittest.mock import patch
        batches = jobs.plan_batches("fake", 1, 32, 32)
        source = {"video_id": "fake", "source_manifest": "fake/input.json",
            "source_manifest_sha256": "manifest", "source_sha256": "sourcehash",
            "source_bytes": 1, "duration_seconds": 1.0, "width": 32, "height": 32,
            "source_path_local": "/fake/video", "crop_last_rows": 0, "batches": batches}
        model_specs = {"sr": {"path": "weights/sr", "sha256": "c"*64}}
        patches = [patch.object(jobs, "_load_source_specs", return_value=[source]),
                   patch.object(jobs, "_load_model_specs", return_value=model_specs),
                   patch.object(jobs, "_collect_code_digests", return_value={"runner": "d"*64}),
                   patch.object(jobs, "_source_tree_prerequisites", return_value={}),
                   patch.object(jobs, "_git_revision", return_value="revision")]
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            repo_root = Path(temporary)/"repo"
            output = repo_root/"outputs"/"dispatcher"
            def interrupted_sampler(video, destination, **kwargs):
                destination.mkdir(parents=True)
                (destination/"partial.ppm").write_bytes(b"partial")
                raise RuntimeError("simulated interruption")

            def good_sampler(video, destination, **kwargs):
                destination.mkdir(parents=True)
                rgb = destination/"frame-000.ppm"
                rgb.write_bytes(b"P6\n32 32\n255\n" + b"\x05\x06\x07"*(32*32))
                frame = {"index": 0, "requested_seconds": 0.0, "pts_seconds": 0.0,
                         "rgb_file": rgb.name, "rgb_sha256": jobs.sha256_file(rgb),
                         "width": 32, "height": 32}
                manifest = {"source_sha256": "sourcehash", "frames": [frame]}
                (destination/"input.json").write_text(json.dumps(manifest))
                return {"source_sha256": "sourcehash"}

            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                first = jobs.run(output, prepare=True, sampler=interrupted_sampler,
                    disk_free_bytes=30*2**30, available_ram_bytes=3*2**30,
                    repo_root=repo_root)
                self.assertTrue((output/"plan.json").is_file())
                self.assertEqual(first["batches"][0]["status"], "failed")
                self.assertFalse((output/"batches"/batches[0]["batch_id"]).exists())
                second = jobs.run(output, prepare=True, sampler=good_sampler,
                    disk_free_bytes=30*2**30, available_ram_bytes=3*2**30,
                    repo_root=repo_root)
                self.assertEqual(second["batches"][0]["status"], "prepared", second["batches"][0])
                self.assertEqual(second["preparation_state"], "prepared")
                manifest = output/"batches"/batches[0]["batch_id"]/"input.json"
                manifest.write_text(manifest.read_text()+" ")
                with self.assertRaisesRegex(ValueError, "hash differs from the frozen plan"):
                    jobs.run(output, prepare=False, repo_root=repo_root)

    def test_exdev_fails_closed_without_replacing_plan(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            plan_path = Path(temporary)/"plan.json"
            plan_path.write_text('{"last_valid":true}\n')
            with patch.object(jobs.os, "replace",
                              side_effect=OSError(errno.EXDEV, "cross-device link")):
                with self.assertRaises(OSError) as caught:
                    jobs._atomic_json(plan_path, {"new": True})
            self.assertEqual(caught.exception.errno, errno.EXDEV)
            self.assertEqual(plan_path.read_text(), '{"last_valid":true}\n')
            self.assertEqual(list(Path(temporary).glob("long-video-plan-*.json")), [])

    def test_batch_promotion_exdev_leaves_no_visible_partial_batch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            root = Path(temporary)
            batch = jobs.plan_batches("fake", 1, 32, 32)[0]
            batch_dir = root/"batches"/batch["batch_id"]
            source = {"source_path_local": "/fake/video", "source_sha256": "sourcehash",
                      "duration_seconds": 1.0}

            def good_sampler(video, destination, **kwargs):
                destination.mkdir(parents=True)
                rgb = destination/"frame-000.ppm"
                rgb.write_bytes(b"P6\n32 32\n255\n" + b"\x01\x02\x03"*(32*32))
                frame = {"index": 0, "requested_seconds": 0.0, "pts_seconds": 0.0,
                         "rgb_file": rgb.name, "rgb_sha256": jobs.sha256_file(rgb),
                         "width": 32, "height": 32}
                (destination/"input.json").write_text(json.dumps(
                    {"source_sha256": "sourcehash", "frames": [frame]}))
                return {"source_sha256": "sourcehash"}

            with patch.object(jobs.os, "rename",
                              side_effect=OSError(errno.EXDEV, "cross-device link")):
                with self.assertRaisesRegex(OSError, "remains uncommitted"):
                    jobs._materialize_batch(batch_dir, batch, source, sampler=good_sampler)
            self.assertFalse(batch_dir.exists())
            self.assertEqual(list((root/"batches").iterdir()), [])


if __name__ == "__main__":
    unittest.main()

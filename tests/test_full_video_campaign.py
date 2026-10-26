"""CPU-only campaign boundary, request identity and failure cleanup tests."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import full_video_campaign as campaign
import run_full_video_job as worker


class FullVideoTests(unittest.TestCase):
    def job(self):
        model, block = "sr", 0
        job = {"model": model, "block": block, "profile": "dronecrowd-full0269-sr-v1",
            "input_manifest": "outputs/video-completo-0269-20261025/inputs/block-000.json",
            "input_sha256": "a"*64, "checkpoint": campaign.previous.CHECKPOINTS[model]["path"],
            "checkpoint_sha256": campaign.previous.CHECKPOINTS[model]["sha256"],
            "output_dir": "outputs/video-completo-0269-20261025/results/sr/block-000",
            "rgb_cache": str(campaign.CACHE/"block-000"), "resources": campaign.RESOURCES}
        plan = {"jobs": [job], "code_digests": {"fixture": "b"*64}}
        job["job_id"] = campaign.previous.canonical_hash({"model": model,
            "input_sha256": job["input_sha256"], "checkpoint_sha256": job["checkpoint_sha256"],
            "code_digests": plan["code_digests"]})
        return plan, job

    def test_frozen_request_rejects_resource_path_profile_and_checkpoint_changes(self):
        plan, job = self.job()
        self.assertEqual(worker.select_job(plan, job["job_id"]), job)
        for field, value in [("resources", {}), ("rgb_cache", "/unexpected"),
                             ("checkpoint_sha256", "c"*64), ("profile", "other"),
                             ("output_dir", "outputs/other"), ("block", 36)]:
            changed = copy.deepcopy(plan)
            changed["jobs"][0][field] = value
            with self.assertRaises(ValueError):
                worker.select_job(changed, job["job_id"])

    def test_completion_rejects_wrong_original_pts_or_missing_tail(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp:
            base = Path(temp)
            frames = [{"source_ordinal": i, "pts": i*1001, "time_base": "1/60000"} for i in range(16)]
            manifest = base/"input.json"
            manifest.write_text(json.dumps({"source_sha256": campaign.SOURCE_HASH, "frames": frames}))
            report = {"source_sha256": campaign.SOURCE_HASH, "results": copy.deepcopy(frames)}
            (base/"report.json").write_text(json.dumps(report))
            with patch.object(worker.bounded, "completed_output", return_value=True):
                self.assertTrue(worker.completed(base, {"block": 0}, manifest, {}))
                report["results"][5]["pts"] += 1
                (base/"report.json").write_text(json.dumps(report))
                self.assertFalse(worker.completed(base, {"block": 0}, manifest, {}))
                self.assertFalse(worker.completed(base, {"block": 35}, manifest, {}))

    def coordinate_fixture(self, temp, failing=False):
        root = Path(temp)
        base, cache = root/"outputs/campaign", root/"cache"
        base.mkdir(parents=True)
        (cache/"block-000").mkdir(parents=True)
        frames = [{"rgb_file": f"frame-{i:03d}.ppm"} for i in range(16)]
        manifest = root/"input.json"
        manifest.write_text(json.dumps({"frames": frames}))
        for frame in frames:
            (cache/"block-000"/frame["rgb_file"]).write_bytes(b"test RGB cache")
        jobs = [{"job_id": str(i)*64, "model": model, "block": 0,
                 "profile": "fixed-"+model, "input_manifest": "input.json",
                 "output_dir": "result-"+model} for i, model in enumerate(campaign.MODELS, 1)]
        events = []
        cancelled = set()
        def queue(_dispatcher, action, *args, **kwargs):
            events.append((action, args))
            if action == "submit":
                return {"key": "DroneCrowd:"+args[-1]}
            if action == "cancel":
                cancelled.add(args[-1])
                return {"status": "requested"}
            return [{"key": "DroneCrowd:"+j["job_id"],
                     "state": ("failed" if index == 0 else
                               "cancelled" if "DroneCrowd:"+j["job_id"] in cancelled else "queued")
                              if failing else "completed"} for index, j in enumerate(jobs)]
        contexts = [patch.object(campaign, "ROOT", root), patch.object(campaign, "BASE", base),
            patch.object(campaign, "CACHE", cache), patch.object(campaign, "TEMP", root),
            patch.object(campaign, "preflight", return_value={}),
            patch.object(campaign, "validate_cache", return_value=cache/"block-000"),
            patch.object(campaign, "queue_call", side_effect=queue),
            patch.object(campaign, "validate_anchor", return_value={"anchors": 1}),
            patch.object(campaign.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()),
            patch.object(worker, "load_plan", return_value={"jobs": jobs}),
            patch.object(worker, "completed", side_effect=[False]*3+[True]*3)]
        return contexts, events, base, cache

    def test_pilot_submits_three_requests_then_validates_and_cleans_only_cache(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp, ExitStack() as stack:
            contexts, events, base, cache = self.coordinate_fixture(temp)
            for context in contexts:
                stack.enter_context(context)
            campaign.coordinate(Path("dispatcher"), Path("quota"), "a"*64, True)
            self.assertEqual(sum(action == "submit" for action, _ in events), 3)
            self.assertFalse((cache/"block-000").exists())
            self.assertEqual(json.loads((base/"state.json").read_text())["state"], "pilot_complete")

    def test_failed_job_cancels_only_own_requests_and_preserves_cache(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp, ExitStack() as stack:
            contexts, events, base, cache = self.coordinate_fixture(temp, failing=True)
            for context in contexts:
                stack.enter_context(context)
            with self.assertRaises(RuntimeError):
                campaign.coordinate(Path("dispatcher"), Path("quota"), "a"*64, True)
            cancellations = [args[-1] for action, args in events if action == "cancel"]
            self.assertEqual(len(cancellations), 2)
            self.assertTrue(all(key.startswith("DroneCrowd:") for key in cancellations))
            self.assertTrue((cache/"block-000").exists())
            self.assertTrue(json.loads((base/"state.json").read_text())["queue_requests_released"])

    def test_failed_cancellation_does_not_declare_queued_request_released(self):
        plan = {"jobs": [{"job_id": "a"*64}]}
        def queue(_dispatcher, action, *args, **kwargs):
            if action == "cancel":
                raise OSError("cancel unavailable")
            return [{"key": "DroneCrowd:"+"a"*64, "state": "queued"}]
        with patch.object(campaign, "queue_call", side_effect=queue):
            result = campaign.stop_requests(Path("dispatcher"), plan, [], wait_seconds=.01)
        self.assertFalse(result["queue_requests_released"])
        self.assertEqual(len(result["unsettled_keys"]), 1)
        self.assertTrue(result["cancel_errors"])
        self.assertTrue(any("cancel unavailable" in error for error in result["cancel_errors"]))

    def test_restart_adopts_old_running_request_and_never_cancels_neighbor(self):
        plan = {"jobs": [{"job_id": "a"*64}]}
        events = []
        cancelled = False
        def queue(_dispatcher, action, *args, **kwargs):
            nonlocal cancelled
            if action == "cancel":
                events.append(args[-1])
                cancelled = True
                return {}
            return [{"key": "DroneCrowd:"+"a"*64, "state": "cancelled" if cancelled else "running"},
                    {"key": "neighbor:xyz", "state": "running"}]
        with patch.object(campaign, "queue_call", side_effect=queue):
            result = campaign.stop_requests(Path("dispatcher"), plan, [])
        self.assertTrue(result["queue_requests_released"])
        self.assertEqual(events, ["DroneCrowd:"+"a"*64])

    def test_cache_rejects_partial_frames_and_root_symlink(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp:
            root = Path(temp)
            cache = root/"cache"
            block = cache/"block-000"
            block.mkdir(parents=True)
            frame = {"index": 0, "rgb_file": "frame-000.ppm", "rgb_sha256": "a"*64}
            (block/"frame-000.ppm").write_bytes(b"incomplete")
            with patch.object(campaign, "TEMP", root), patch.object(campaign, "CACHE", cache):
                with self.assertRaises(ValueError):
                    campaign.validate_cache(0, {"frames": [frame]})
                other = root/"linked-cache"
                other.symlink_to(cache, target_is_directory=True)
                with patch.object(campaign, "CACHE", other):
                    with self.assertRaises(ValueError):
                        campaign.cache_path(0)

    def test_dispatcher_identity_mismatch_fails_before_scope_or_inference(self):
        plan, job = self.job()
        good = {"NEXO_GPU_REQUEST_ID": job["job_id"], "NEXO_GPU_PROJECT": "DroneCrowd",
                "NEXO_GPU_PROFILE": job["profile"]}
        for key, value in [("NEXO_GPU_PROJECT", "neighbor"), ("NEXO_GPU_PROFILE", "other"),
                           ("NEXO_GPU_REQUEST_ID", "bad-id")]:
            with patch.dict(os.environ, {**good, key: value}), \
                 patch.object(sys, "argv", ["worker", "--plan", "ignored", "--plan-sha256", "a"*64, "--model", "sr"]), \
                 patch.object(worker, "load_plan", return_value=plan), \
                 patch.object(worker.bounded, "run_job") as run, \
                 patch.object(worker.bounded, "write_preflight_failure", return_value=Path("evidence.json")):
                self.assertEqual(worker.main(), 1)
                run.assert_not_called()

    def test_resume_only_submits_remaining_model_of_block(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp, ExitStack() as stack:
            contexts, events, base, cache = self.coordinate_fixture(temp)
            for context in contexts:
                stack.enter_context(context)
            stack.enter_context(patch.object(worker, "completed", side_effect=[True, True, False]+[True]*3))
            campaign.coordinate(Path("dispatcher"), Path("quota"), "a"*64, True)
            self.assertEqual(sum(action == "submit" for action, _ in events), 1)

    def test_cancel_waits_for_recovering_request_to_be_terminal(self):
        plan = {"jobs": [{"job_id": "a"*64}]}
        states = iter(["recovering", "recovering", "cancelled"])
        def queue(_dispatcher, action, *args, **kwargs):
            if action == "cancel":
                return {}
            return [{"key": "DroneCrowd:"+"a"*64, "state": next(states)}]
        with patch.object(campaign, "queue_call", side_effect=queue), patch.object(campaign.time, "sleep") as sleep:
            result = campaign.stop_requests(Path("dispatcher"), plan, [])
        self.assertTrue(result["queue_requests_released"])
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()

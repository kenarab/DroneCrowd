"""CPU-only identity, tails, migration and cancellation boundaries."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import night_video_campaign as night


class NightTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [{"job_id": str(i+1)*64, "model": m, "block": 4,
                      "profile": night.profile("0269", m)} for i, m in enumerate(night.MODELS)]
        self.plan = {"jobs": self.jobs}
        self.data = {"project": "DroneCrowd", "plan_sha256": night.ORIGINAL_PLAN_SHA,
                     "replacements": []}
        for j in self.jobs:
            old = j["job_id"]
            self.data["replacements"].append({"scientific_job_id": old,
                "request_id": hashlib.sha256(("supersedes:DroneCrowd:"+old).encode()).hexdigest(),
                "original_receipt": {"key": "DroneCrowd:"+old, "project": "DroneCrowd",
                    "request_id": old, "profile": j["profile"], "state": "cancelled",
                    "started": None, "launcher_pid": None, "launcher_start": None}})

    def validate(self, data=None):
        def select(p, job_id):
            return next(j for j in p["jobs"] if j["job_id"] == job_id)
        with patch.object(night.old_worker, "select_job", side_effect=select):
            return night.validate_migration(data or self.data, self.plan)

    def test_only_new_execution_id_maps_to_original_scientific_job(self):
        mapping = self.validate()
        for old, new in mapping.items():
            self.assertEqual(night.scientific_id(new, mapping, self.plan), old)
            with self.assertRaises(ValueError):
                night.scientific_id(old, mapping, self.plan)
        with self.assertRaises(ValueError):
            night.scientific_id("a"*64, mapping, self.plan)

    def test_completed_started_or_neighbor_replacement_rejected(self):
        for key, value in [("state", "completed"), ("started", 1.0),
                           ("project", "neighbor"), ("launcher_pid", 123),
                           ("launcher_start", "123"), ("profile", "other")]:
            data = copy.deepcopy(self.data)
            data["replacements"][0]["original_receipt"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(data)
        data = copy.deepcopy(self.data)
        del data["replacements"][0]["original_receipt"]["started"]
        with self.assertRaises(ValueError):
            self.validate(data)

    def test_duplicate_or_arbitrary_alias_and_wrong_block_rejected(self):
        for mutate in [lambda d: d["replacements"].__setitem__(1, copy.deepcopy(d["replacements"][0])),
                       lambda d: d["replacements"][0].__setitem__("request_id", "a"*64)]:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(ValueError):
                self.validate(data)
        self.jobs[0]["block"] = 0
        with self.assertRaises(ValueError):
            self.validate()

    def test_completed_scientific_id_is_unchanged(self):
        mapping = self.validate()
        self.plan["jobs"].append({"job_id": "f"*64})
        self.assertEqual(night.scientific_id("f"*64, mapping, self.plan), "f"*64)

    def test_tails_and_trim_crop_are_fixed(self):
        self.assertEqual([night.blocks(s) for s in ("0265", "0269", "trim")], [86, 36, 42])
        self.assertEqual([night.SOURCES[s]["count"]-(night.blocks(s)-1)*16
                          for s in ("0265", "0269", "trim")], [10, 3, 3])
        self.assertEqual(night.header("trim"), b"P6\n960 528\n255\n")

    def test_partial_or_symlink_cache_preserved(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            root = Path(tmp)
            cache = root/"cache"
            block = cache/"block-000"
            block.mkdir(parents=True)
            file = block/"frame-000.ppm"
            file.write_bytes(b"partial")
            with patch.object(night, "TEMP", root), patch.object(night, "cache_root", return_value=cache):
                with self.assertRaises(ValueError):
                    night.validate_cache("trim", 0, {"frames": [{"rgb_file": file.name}]})
                self.assertEqual(file.read_bytes(), b"partial")
                file.unlink()
                file.symlink_to(root/"outside")
                with self.assertRaises(ValueError):
                    night.validate_cache("trim", 0, {"frames": [{"rgb_file": file.name}]})

    def test_identity_failure_precedes_gpu_scope(self):
        with patch.object(night, "load_plan", return_value=self.plan), \
             patch.object(night, "migration", return_value=self.validate()), \
             patch.object(night.bounded, "run_job") as run, \
             patch.dict(os.environ, {"NEXO_GPU_REQUEST_ID": "a"*64}):
            with self.assertRaises(ValueError):
                night.work("0269", night.ORIGINAL_PLAN_SHA, "sr", "b"*64)
            run.assert_not_called()

    def test_replacement_failure_cleanup_uses_new_ids_and_preserves_neighbor(self):
        mapping = self.validate()
        scheduler = {"jobs": [{"job_id": new} for new in mapping.values()]}
        calls = []
        def queue(_dispatcher, action, *args, **kwargs):
            if action == "cancel":
                calls.append(args[-1])
                return {}
            return ([{"key": "DroneCrowd:"+new, "state": "cancelled" if calls else "queued"}
                     for new in mapping.values()]+[{"key": "neighbor:job", "state": "running"}])
        with patch.object(night.legacy, "queue_call", side_effect=queue):
            result = night.legacy.stop_requests(Path("dispatcher"), scheduler, [])
        self.assertTrue(result["queue_requests_released"])
        self.assertEqual(set(calls), {"DroneCrowd:"+new for new in mapping.values()})

    def test_coordinator_resumes_only_missing_replacement_using_actual_receipt(self):
        self.coordinate_case()

    def test_coordinator_missing_supersedes_stops_and_cleans_execution_ids(self):
        self.coordinate_case(bad_supersedes=True)

    def coordinate_case(self, bad_supersedes=False):
        mapping = self.validate()
        done = False
        calls = []
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp, ExitStack() as stack:
            root = Path(tmp)
            output = root/"out"
            output.mkdir()
            jobs = copy.deepcopy(self.jobs)
            for job in jobs:
                job.update(output_dir=f"results/{job['model']}", input_manifest="input.json")
            (root/"input.json").write_text(json.dumps({"frames": []}))
            (output/"night-migration.json").write_text("{}")
            plan = {"jobs": jobs}
            def complete(source, dest, job, manifest, p):
                return done or job["model"] != "mpcount"
            def queue(dispatcher, action, *args, **kw):
                nonlocal done
                calls.append((action, args))
                if action == "cancel":
                    return {}
                done = True
                return [{"key": "DroneCrowd:"+mapping[j["job_id"]], "project": "DroneCrowd",
                    "profile": j["profile"], "state": "completed", "request_id": mapping[j["job_id"]],
                    "supersedes": "wrong" if bad_supersedes else "DroneCrowd:"+j["job_id"]} for j in jobs]
            for obj, name, kwargs in [
                (night, "ROOT", {"new": root}), (night, "base", {"return_value": output}),
                (night, "load_plan", {"return_value": plan}), (night, "blocks", {"return_value": 5}),
                (night, "migration", {"return_value": mapping}), (night, "validate_reuse", {}),
                (night, "completed", {"side_effect": complete}),
                (night, "cache_path", {"side_effect": lambda s, b: root/f"cache-{b}"}),
                (night, "validate_cache", {}), (night, "summarize", {}),
                (night.bounded, "verify_code_digests", {}), (night.legacy, "preflight", {}),
                (night.legacy, "queue_call", {"side_effect": queue}),
                (night.subprocess, "run", {"return_value": subprocess_result()}),
            ]:
                stack.enter_context(patch.object(obj, name, **kwargs))
            if bad_supersedes:
                with self.assertRaises(ValueError):
                    night.coordinate("0269", "a"*64, "b"*64, Path("dispatcher"), Path("quota"))
                state = json.loads((output/"state.json").read_text())
                self.assertEqual(state["state"], "stopped")
                self.assertTrue(state["queue_requests_released"])
            else:
                night.coordinate("0269", "a"*64, "b"*64, Path("dispatcher"), Path("quota"))
                row = json.loads((output/"night-receipt-mpcount-004.json").read_text())
                self.assertEqual(row["request_id"], mapping[jobs[2]["job_id"]])
                self.assertEqual(row["supersedes"], "DroneCrowd:"+jobs[2]["job_id"])
            self.assertFalse(any(action == "submit" for action, args in calls))

    def test_reuse_missing_or_duplicate_jobs_rejected_before_file_reads(self):
        plan = {"jobs": [{"job_id": f"{i:064x}", "block": i//3} for i in range(12)]}
        for entries in [[], [{"scientific_job_id": f"{i:064x}"} for i in range(11)],
                        [{"scientific_job_id": "0"*64}]*12]:
            with self.assertRaises(ValueError):
                night.validate_reuse({"reused_results": entries}, plan)

    def test_cli_preserves_preflight_failure_evidence_without_launch(self):
        with patch.object(sys, "argv", ["night", "work", "--source", "0269", "--plan-sha256", "a"*64]), \
             patch.object(night, "work", side_effect=ValueError("wrong identity")), \
             patch.object(night.bounded, "write_preflight_failure", return_value=Path("evidence.json")) as evidence:
            self.assertEqual(night.main(), 1)
            evidence.assert_called_once()

    def test_pinned_reuse_rejects_modified_report_and_missing_map(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            root = Path(tmp)
            (root/"input.json").write_text("input")
            jobs, ledger = [], []
            for i in range(12):
                model, block = night.MODELS[i%3], i//3
                job_id = f"{i:064x}"
                output = root/f"results/{i}"
                output.mkdir(parents=True)
                frames = [{"source_ordinal": block*16+k, "density_file": f"density-{k:03d}.npy",
                           "density_sha256": "a"*64} for k in range(16)]
                report = output/"report.json"
                report.write_text(json.dumps({"results": frames}))
                job = {"job_id": job_id, "model": model, "block": block,
                    "output_dir": f"results/{i}", "input_manifest": "input.json",
                    "input_sha256": night.bounded.sha256_file(root/"input.json"),
                    "checkpoint_sha256": "b"*64, "profile": night.profile("0269", model)}
                jobs.append(job)
                ledger.append({"scientific_job_id": job_id, "input_sha256": job["input_sha256"],
                    "checkpoint_sha256": "b"*64, "report_sha256": night.bounded.sha256_file(report),
                    "maps": [{"file": f["density_file"], "sha256": f["density_sha256"],
                              "source_ordinal": f["source_ordinal"]} for f in frames],
                    "original_receipt": {"key": "DroneCrowd:"+job_id, "request_id": job_id,
                        "project": "DroneCrowd", "profile": job["profile"], "state": "completed",
                        "exit_code": 0, "started": 1.0}})
            data, plan = {"reused_results": ledger}, {"jobs": jobs}
            with patch.object(night, "ROOT", root), patch.object(night, "completed", return_value=True):
                night.validate_reuse(data, plan)
                bad = copy.deepcopy(data)
                bad["reused_results"][0]["maps"].pop()
                with self.assertRaises(ValueError):
                    night.validate_reuse(bad, plan)
                (root/"results/0/report.json").write_text('{"results": []}')
                with self.assertRaises(ValueError):
                    night.validate_reuse(data, plan)

    def test_existing_csv_cannot_silently_override_verified_reports(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            root = Path(tmp)
            (root/"plan.json").write_text("{}")
            (root/"result").mkdir()
            (root/"result/report.json").write_text("{}")
            plan = {"jobs": [{"output_dir": "result"}]}
            rows = [{"source": "0269", "source_ordinal": 0, "pts": 0, "time_base": "1/60000",
                     "pts_seconds": 0.0, "rgb_sha256": "a"*64, "sr": 1.0, "clip-ebc": 2.0, "mpcount": 3.0}]
            with patch.object(night, "ROOT", root), patch.object(night, "base", return_value=root), \
                 patch.object(night, "prediction_rows", return_value=rows):
                night.summarize("0269", plan)
                night.summarize("0269", plan)
                path = root/"counts.csv"
                path.write_bytes(path.read_bytes().replace(b"1.0,2.0,3.0", b"999,2.0,3.0"))
                with self.assertRaises(ValueError):
                    night.summarize("0269", plan)


def subprocess_result():
    import subprocess
    return subprocess.CompletedProcess([], 0)


if __name__ == "__main__":
    unittest.main()
